//! Z-Image Studio - Low Memory Mode
//!
//! This GUI uses process isolation to run on 8GB GPUs:
//! 1. Spawns compute_embedding.exe to encode text (uses ~4GB, then exits)
//! 2. Loads transformer + autoencoder in main process (~6GB)
//! 3. Generates image from cached embedding
//!
//! Build: cargo build --release --no-default-features --features "native-cuda,egui" --bin gui_lowmem

use std::path::PathBuf;
use std::process::Command;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

use eframe::egui;

#[cfg(feature = "native-cuda")]
use burn::backend::{Cuda, cuda::CudaDevice};

#[cfg(feature = "native-cuda")]
type Backend = Cuda<f32, i32>;

#[cfg(feature = "native-cuda")]
type Device = CudaDevice;

#[cfg(feature = "native-cuda")]
fn create_device() -> Device {
    CudaDevice::new(0)
}

const BACKEND_NAME: &str = "CUDA (Native) - Low Memory Mode";

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([900.0, 700.0])
            .with_min_inner_size([600.0, 400.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Z-Image Studio (Low Memory)",
        options,
        Box::new(|cc| Ok(Box::new(ZImageApp::new(cc)))),
    )
}

enum WorkerMessage {
    Log(String),
    EmbeddingComplete(Result<PathBuf, String>),
    GenerationComplete(Result<PathBuf, String>),
}

#[derive(Clone, Copy, PartialEq)]
enum AppState {
    Idle,
    ComputingEmbedding,
    LoadingModels,
    Generating,
}

struct ZImageApp {
    // Model directory
    model_dir: String,

    // Generation settings
    prompt: String,
    width: i32,
    height: i32,
    steps: i32,

    // State
    state: AppState,
    current_embedding: Option<PathBuf>,
    generated_image: Option<egui::TextureHandle>,
    last_image_path: Option<PathBuf>,

    // Async
    tx: Sender<WorkerMessage>,
    rx: Receiver<WorkerMessage>,

    // Logs
    logs: Vec<String>,

    // History
    history: Vec<HistoryItem>,
}

#[derive(Clone)]
struct HistoryItem {
    prompt: String,
    embedding_path: PathBuf,
    image_path: Option<PathBuf>,
}

impl ZImageApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let (tx, rx) = mpsc::channel();

        Self {
            model_dir: "D:/burn_models/z-image".to_string(),
            prompt: "A beautiful sunset over mountains".to_string(),
            width: 256,
            height: 256,
            steps: 4,
            state: AppState::Idle,
            current_embedding: None,
            generated_image: None,
            last_image_path: None,
            tx,
            rx,
            logs: vec!["Z-Image Studio (Low Memory Mode) started".to_string()],
            history: Vec::new(),
        }
    }

    fn log(&mut self, msg: &str) {
        let timestamp = chrono::Local::now().format("%H:%M:%S").to_string();
        self.logs.push(format!("[{}] {}", timestamp, msg));
        if self.logs.len() > 100 {
            self.logs.remove(0);
        }
    }

    fn generate(&mut self) {
        if self.prompt.trim().is_empty() {
            self.log("Error: Prompt is empty");
            return;
        }

        self.state = AppState::ComputingEmbedding;
        self.log(&format!("Starting generation: {}", self.prompt));

        // Create embeddings directory
        let embeddings_dir = PathBuf::from(&self.model_dir).join("embeddings");
        let _ = std::fs::create_dir_all(&embeddings_dir);

        // Generate unique filename for this embedding
        let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S").to_string();
        let embedding_path = embeddings_dir.join(format!("{}.zemb", timestamp));

        let tx = self.tx.clone();
        let model_dir = self.model_dir.clone();
        let prompt = self.prompt.clone();
        let emb_path = embedding_path.clone();

        // Spawn compute_embedding as separate process
        thread::spawn(move || {
            let _ = tx.send(WorkerMessage::Log("Spawning embedding process...".to_string()));

            // Find compute_embedding.exe in same directory as this exe
            let exe_dir = std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|p| p.to_path_buf()))
                .unwrap_or_else(|| PathBuf::from("."));

            let compute_exe = exe_dir.join("compute_embedding.exe");

            let _ = tx.send(WorkerMessage::Log(format!("Running: {:?}", compute_exe)));

            let result = Command::new(&compute_exe)
                .arg("--model-dir").arg(&model_dir)
                .arg("--prompt").arg(&prompt)
                .arg("--output").arg(&emb_path)
                .arg("--force")
                .output();

            match result {
                Ok(output) => {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    for line in stderr.lines() {
                        if !line.trim().is_empty() && !line.contains("Scalar(Float") {
                            let _ = tx.send(WorkerMessage::Log(line.to_string()));
                        }
                    }

                    if output.status.success() && emb_path.exists() {
                        let _ = tx.send(WorkerMessage::EmbeddingComplete(Ok(emb_path)));
                    } else {
                        let _ = tx.send(WorkerMessage::EmbeddingComplete(
                            Err(format!("Process failed: {}", stderr))
                        ));
                    }
                }
                Err(e) => {
                    let _ = tx.send(WorkerMessage::EmbeddingComplete(
                        Err(format!("Failed to spawn process: {}", e))
                    ));
                }
            }
        });
    }

    #[cfg(feature = "native-cuda")]
    fn generate_from_embedding(&mut self, embedding_path: PathBuf) {
        self.state = AppState::LoadingModels;
        self.log("Loading transformer and autoencoder...");

        let tx = self.tx.clone();
        let model_dir = self.model_dir.clone();
        let width = self.width as usize;
        let height = self.height as usize;
        let steps = self.steps as usize;

        // Generate output path
        let outputs_dir = PathBuf::from(&model_dir).join("outputs");
        let _ = std::fs::create_dir_all(&outputs_dir);
        let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S").to_string();
        let output_path = outputs_dir.join(format!("{}_{}x{}.png", timestamp, width, height));

        thread::spawn(move || {
            use z_image::{GenerateWithEmbeddingOpts, modules::ae::AutoEncoderConfig, modules::transformer::ZImageModelConfig};
            use burn::Tensor;
            use std::io::Read;

            let device = create_device();

            // Enable memory optimizations for 8GB VRAM
            // Head slicing: process 4 heads at a time (24 heads / 4 = 6 passes)
            z_image::set_attention_slice_size(4);
            // Sequence chunking: chunk 2304 tokens into 512-token pieces (4-5 passes)
            // This reduces attention memory from 500MB to ~125MB
            z_image::set_attention_seq_chunk_size(512);
            let _ = tx.send(WorkerMessage::Log("Memory optimizations enabled (head_slice=4, seq_chunk=512)".to_string()));

            // Load embedding from file
            let _ = tx.send(WorkerMessage::Log("Loading embedding...".to_string()));

            let embedding = match load_embedding(&embedding_path, &device) {
                Ok(e) => e,
                Err(e) => {
                    let _ = tx.send(WorkerMessage::GenerationComplete(Err(e)));
                    return;
                }
            };

            let _ = tx.send(WorkerMessage::Log(format!("Embedding loaded: {:?}", embedding.dims())));

            // Load transformer
            let transformer_path = PathBuf::from(&model_dir).join("z_image_turbo_q8.bpk");
            let _ = tx.send(WorkerMessage::Log(format!("Loading transformer...")));

            let mut transformer: z_image::modules::transformer::ZImageModel<Backend> =
                ZImageModelConfig::default().init(&device);
            if let Err(e) = transformer.load_weights(&transformer_path) {
                let _ = tx.send(WorkerMessage::GenerationComplete(
                    Err(format!("Failed to load transformer: {:?}", e))
                ));
                return;
            }
            let _ = tx.send(WorkerMessage::Log("Transformer loaded (~6GB VRAM)".to_string()));

            // Load autoencoder
            let ae_path = PathBuf::from(&model_dir).join("ae.bpk");
            let _ = tx.send(WorkerMessage::Log("Loading autoencoder...".to_string()));

            let mut ae: z_image::modules::ae::AutoEncoder<Backend> =
                AutoEncoderConfig::flux_ae().init(&device);
            if let Err(e) = ae.load_weights(&ae_path) {
                let _ = tx.send(WorkerMessage::GenerationComplete(
                    Err(format!("Failed to load autoencoder: {:?}", e))
                ));
                return;
            }
            let _ = tx.send(WorkerMessage::Log("Autoencoder loaded".to_string()));

            // Generate
            let _ = tx.send(WorkerMessage::Log(format!("Generating {}x{} image with {} steps...", width, height, steps)));

            let opts = GenerateWithEmbeddingOpts {
                width,
                height,
                num_inference_steps: Some(steps),
                seed: None,
            };

            match z_image::generate_with_embedding(embedding, &output_path, &opts, &ae, &transformer, &device) {
                Ok(()) => {
                    let _ = tx.send(WorkerMessage::GenerationComplete(Ok(output_path)));
                }
                Err(e) => {
                    let _ = tx.send(WorkerMessage::GenerationComplete(
                        Err(format!("Generation failed: {:?}", e))
                    ));
                }
            }
        });
    }

    fn process_messages(&mut self, ctx: &egui::Context) {
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                WorkerMessage::Log(text) => self.log(&text),
                WorkerMessage::EmbeddingComplete(result) => {
                    match result {
                        Ok(path) => {
                            self.log(&format!("Embedding saved: {:?}", path));
                            self.current_embedding = Some(path.clone());

                            // Add to history
                            self.history.push(HistoryItem {
                                prompt: self.prompt.clone(),
                                embedding_path: path.clone(),
                                image_path: None,
                            });

                            // Now generate from embedding
                            #[cfg(feature = "native-cuda")]
                            self.generate_from_embedding(path);
                        }
                        Err(e) => {
                            self.log(&format!("Embedding failed: {}", e));
                            self.state = AppState::Idle;
                        }
                    }
                }
                WorkerMessage::GenerationComplete(result) => {
                    self.state = AppState::Idle;
                    match result {
                        Ok(path) => {
                            self.log(&format!("Image saved: {:?}", path));
                            self.last_image_path = Some(path.clone());

                            // Update history
                            if let Some(item) = self.history.last_mut() {
                                item.image_path = Some(path.clone());
                            }

                            // Load image for display
                            if let Ok(image_data) = std::fs::read(&path) {
                                if let Ok(image) = image::load_from_memory(&image_data) {
                                    let size = [image.width() as _, image.height() as _];
                                    let image_buffer = image.to_rgba8();
                                    let pixels = image_buffer.as_flat_samples();
                                    let color_image = egui::ColorImage::from_rgba_unmultiplied(
                                        size,
                                        pixels.as_slice(),
                                    );
                                    self.generated_image = Some(ctx.load_texture(
                                        "generated",
                                        color_image,
                                        egui::TextureOptions::default(),
                                    ));
                                }
                            }
                        }
                        Err(e) => {
                            self.log(&format!("Generation failed: {}", e));
                        }
                    }
                }
            }
        }
    }
}

#[cfg(feature = "native-cuda")]
fn load_embedding(path: &PathBuf, device: &Device) -> Result<burn::Tensor<Backend, 3>, String> {
    use std::io::Read;
    use burn::Tensor;

    let mut file = std::fs::File::open(path)
        .map_err(|e| format!("Failed to open: {}", e))?;

    let mut magic = [0u8; 4];
    file.read_exact(&mut magic).map_err(|e| format!("Read error: {}", e))?;
    if &magic != b"ZEMB" {
        return Err("Invalid embedding format".to_string());
    }

    let mut buf = [0u8; 4];
    file.read_exact(&mut buf).map_err(|e| format!("Read error: {}", e))?;
    let _version = u32::from_le_bytes(buf);

    file.read_exact(&mut buf).map_err(|e| format!("Read error: {}", e))?;
    let dim0 = u32::from_le_bytes(buf) as usize;

    file.read_exact(&mut buf).map_err(|e| format!("Read error: {}", e))?;
    let dim1 = u32::from_le_bytes(buf) as usize;

    file.read_exact(&mut buf).map_err(|e| format!("Read error: {}", e))?;
    let dim2 = u32::from_le_bytes(buf) as usize;

    let num_floats = dim0 * dim1 * dim2;
    let mut floats = Vec::with_capacity(num_floats);

    for _ in 0..num_floats {
        file.read_exact(&mut buf).map_err(|e| format!("Read error: {}", e))?;
        floats.push(f32::from_le_bytes(buf));
    }

    let tensor: Tensor<Backend, 1> = Tensor::from_floats(floats.as_slice(), device);
    Ok(tensor.reshape([dim0, dim1, dim2]))
}

impl eframe::App for ZImageApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.process_messages(ctx);

        // Header
        egui::TopBottomPanel::top("header").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Z-Image Studio");
                ui.separator();
                ui.label(BACKEND_NAME);

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let status = match self.state {
                        AppState::Idle => ("Ready", egui::Color32::GREEN),
                        AppState::ComputingEmbedding => ("Computing embedding...", egui::Color32::YELLOW),
                        AppState::LoadingModels => ("Loading models...", egui::Color32::YELLOW),
                        AppState::Generating => ("Generating...", egui::Color32::YELLOW),
                    };
                    ui.colored_label(status.1, status.0);
                    if self.state != AppState::Idle {
                        ui.spinner();
                    }
                });
            });
        });

        // Status bar
        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("8GB GPU Mode: Process isolation enabled");
                ui.separator();
                if let Some(path) = &self.last_image_path {
                    ui.label(format!("Last: {}", path.file_name().unwrap_or_default().to_string_lossy()));
                }
            });
        });

        // Right panel: Logs
        egui::SidePanel::right("logs").min_width(250.0).show(ctx, |ui| {
            ui.heading("Logs");
            ui.separator();

            if ui.button("Clear").clicked() {
                self.logs.clear();
            }

            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    for log in &self.logs {
                        let color = if log.contains("Error") || log.contains("failed") {
                            egui::Color32::RED
                        } else if log.contains("loaded") || log.contains("saved") {
                            egui::Color32::GREEN
                        } else {
                            egui::Color32::GRAY
                        };
                        ui.colored_label(color, log);
                    }
                });
        });

        // Main content
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                // Model directory
                ui.horizontal(|ui| {
                    ui.label("Model Directory:");
                    ui.add(egui::TextEdit::singleline(&mut self.model_dir).desired_width(400.0));
                    if ui.button("Browse...").clicked() {
                        if let Some(path) = rfd::FileDialog::new().pick_folder() {
                            self.model_dir = path.to_string_lossy().to_string();
                        }
                    }
                });

                ui.add_space(10.0);

                // Prompt
                ui.label("Prompt:");
                ui.add(
                    egui::TextEdit::multiline(&mut self.prompt)
                        .desired_rows(3)
                        .desired_width(f32::INFINITY)
                );

                ui.add_space(10.0);

                // Settings
                ui.horizontal(|ui| {
                    ui.label("Size:");
                    if ui.selectable_label(self.width == 256, "256x256").clicked() {
                        self.width = 256;
                        self.height = 256;
                    }
                    if ui.selectable_label(self.width == 384, "384x384").clicked() {
                        self.width = 384;
                        self.height = 384;
                    }

                    ui.separator();

                    ui.label("Steps:");
                    ui.add(egui::Slider::new(&mut self.steps, 2..=8));
                });

                ui.add_space(10.0);

                // Generate button
                ui.horizontal(|ui| {
                    let can_generate = self.state == AppState::Idle;
                    if ui.add_enabled(can_generate, egui::Button::new("Generate Image")).clicked() {
                        self.generate();
                    }

                    if self.state != AppState::Idle {
                        ui.spinner();
                        ui.label(match self.state {
                            AppState::ComputingEmbedding => "Step 1/2: Computing text embedding (separate process)...",
                            AppState::LoadingModels => "Step 2/2: Loading transformer...",
                            AppState::Generating => "Step 2/2: Generating image...",
                            _ => "",
                        });
                    }
                });

                ui.add_space(20.0);
                ui.separator();
                ui.add_space(10.0);

                // Generated image
                ui.heading("Generated Image");

                let available = ui.available_size();
                let image_size = available.x.min(512.0);

                egui::Frame::canvas(ui.style()).show(ui, |ui| {
                    ui.set_min_size(egui::vec2(image_size, image_size));

                    if let Some(texture) = &self.generated_image {
                        ui.image(texture);
                    } else {
                        ui.centered_and_justified(|ui| {
                            ui.label("Image will appear here after generation");
                        });
                    }
                });

                // Open image button
                if let Some(path) = &self.last_image_path {
                    ui.add_space(10.0);
                    if ui.button("Open in File Explorer").clicked() {
                        let _ = open::that(path);
                    }
                }

                ui.add_space(20.0);

                // History
                if !self.history.is_empty() {
                    ui.separator();
                    ui.heading("History");

                    for (i, item) in self.history.iter().enumerate().rev().take(5) {
                        ui.horizontal(|ui| {
                            ui.label(format!("{}.", i + 1));
                            ui.label(&item.prompt);
                            if item.image_path.is_some() {
                                ui.colored_label(egui::Color32::GREEN, "Done");
                            }
                        });
                    }
                }
            });
        });

        // Request repaint while working
        if self.state != AppState::Idle {
            ctx.request_repaint();
        }
    }
}
