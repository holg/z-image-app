//! Z-Image Studio - Cross-platform GUI using egui
//!
//! Build for different platforms:
//!   macOS:              cargo build --release --features "egui,metal"
//!   Windows/Linux CUDA: cargo build --release --features "egui,cuda"
//!   Windows/Linux GPU:  cargo build --release --features "egui,vulkan"
//!   CPU only:           cargo build --release --features "egui,cpu"

use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;

use burn::tensor::{Int, Tensor};
use eframe::egui;
use half::bf16;
use qwen3_burn::{Qwen3Config, Qwen3ForCausalLM, Qwen3Model, Qwen3Tokenizer};
use z_image::modules::ae::{AutoEncoder, AutoEncoderConfig};
use z_image::modules::transformer::{ZImageModel, ZImageModelConfig};

// Backend selection based on compile-time features
#[cfg(feature = "metal")]
mod backend {
    use burn::backend::candle::{Candle, CandleDevice};
    use half::bf16;
    pub type Backend = Candle<bf16, i64>;
    pub type Device = CandleDevice;

    pub fn create_device() -> Device {
        CandleDevice::metal(0)
    }

    pub const BACKEND_NAME: &str = "Metal (macOS GPU)";
}

#[cfg(feature = "cuda")]
mod backend {
    use burn::backend::candle::{Candle, CandleDevice};
    use half::bf16;
    pub type Backend = Candle<bf16, i64>;
    pub type Device = CandleDevice;

    pub fn create_device() -> Device {
        CandleDevice::cuda(0)
    }

    pub const BACKEND_NAME: &str = "CUDA (NVIDIA GPU)";
}

#[cfg(feature = "vulkan")]
#[cfg(not(any(feature = "metal", feature = "cuda")))]
mod backend {
    use burn::backend::wgpu::{Wgpu, WgpuDevice};
    pub type Backend = Wgpu<f32, i32>;
    pub type Device = WgpuDevice;

    pub fn create_device() -> Device {
        WgpuDevice::default()
    }

    pub const BACKEND_NAME: &str = "Vulkan (WGPU)";
}

#[cfg(feature = "wgpu-metal")]
#[cfg(not(any(feature = "metal", feature = "cuda", feature = "vulkan")))]
mod backend {
    use burn::backend::wgpu::{Wgpu, WgpuDevice};
    pub type Backend = Wgpu<f32, i32>;
    pub type Device = WgpuDevice;

    pub fn create_device() -> Device {
        WgpuDevice::default()
    }

    pub const BACKEND_NAME: &str = "Metal (WGPU)";
}

#[cfg(feature = "wgpu")]
#[cfg(not(any(feature = "metal", feature = "cuda", feature = "vulkan", feature = "wgpu-metal")))]
mod backend {
    use burn::backend::wgpu::{Wgpu, WgpuDevice};
    pub type Backend = Wgpu<f32, i32>;
    pub type Device = WgpuDevice;

    pub fn create_device() -> Device {
        WgpuDevice::default()
    }

    pub const BACKEND_NAME: &str = "WGPU (Cross-platform GPU)";
}

#[cfg(feature = "cpu")]
#[cfg(not(any(feature = "metal", feature = "cuda", feature = "vulkan", feature = "wgpu")))]
mod backend {
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    pub type Backend = NdArray<f32>;
    pub type Device = NdArrayDevice;

    pub fn create_device() -> Device {
        NdArrayDevice::Cpu
    }

    pub const BACKEND_NAME: &str = "CPU (NdArray)";
}

// Fallback if no backend specified
#[cfg(not(any(feature = "metal", feature = "cuda", feature = "vulkan", feature = "wgpu", feature = "cpu")))]
mod backend {
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    pub type Backend = NdArray<f32>;
    pub type Device = NdArrayDevice;

    pub fn create_device() -> Device {
        NdArrayDevice::Cpu
    }

    pub const BACKEND_NAME: &str = "CPU (fallback)";
}

use backend::{Backend, Device, BACKEND_NAME};

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_min_inner_size([800.0, 600.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Z-Image Studio",
        options,
        Box::new(|cc| Ok(Box::new(ZImageApp::new(cc)))),
    )
}

// Message types for async operations
enum WorkerMessage {
    Log(String),
    ImageGenComplete(Result<PathBuf, String>),
    TextGenComplete(Result<String, String>),
    ModelLoadComplete(ModelType, Result<(), String>),
    DownloadProgress(String, f32),
    DownloadComplete(String, Result<(), String>),
}

#[derive(Clone, Copy, PartialEq)]
enum ModelType {
    Image,
    Text,
}

#[derive(Clone, Copy, PartialEq)]
enum Tab {
    Image,
    History,
    Chat,
    Settings,
    Models,
}

/// A generated image in history
#[derive(Clone)]
struct HistoryItem {
    prompt: String,
    width: i32,
    height: i32,
    steps: i32,
    timestamp: String,
    image_path: PathBuf,
    #[allow(dead_code)]
    generation_time: f32,
}

struct ZImageApp {
    // Current tab
    current_tab: Tab,

    // Model directories
    image_model_dir: String,
    text_model_dir: String,

    // Image generation state
    image_prompt: String,
    image_width: i32,
    image_height: i32,
    generated_image: Option<egui::TextureHandle>,
    is_generating_image: bool,
    image_models_loaded: bool,
    #[allow(dead_code)]
    last_generated_path: Option<PathBuf>,

    // Generation settings
    num_inference_steps: i32,
    use_seed: bool,
    seed: u64,

    // Memory settings
    attention_slice_size: i32,
    low_memory_mode: bool,

    // History
    history: Vec<HistoryItem>,
    selected_history_index: Option<usize>,
    history_dir: PathBuf,

    // Text chat state
    chat_messages: Vec<ChatMessage>,
    chat_input: String,
    is_generating_text: bool,
    text_model_loaded: bool,
    max_tokens: i32,
    temperature: f32,

    // Model download state
    models_base_dir: String,
    download_progress: std::collections::HashMap<String, f32>,
    #[allow(dead_code)]
    download_status: std::collections::HashMap<String, String>,

    // Logs
    logs: Vec<String>,
    show_logs: bool,

    // Device initialized
    device_initialized: bool,

    // Async communication
    tx: Sender<WorkerMessage>,
    rx: Receiver<WorkerMessage>,

    // Shared state for models
    device: Option<Device>,

    // Image models
    image_tokenizer: Arc<Mutex<Option<Qwen3Tokenizer>>>,
    image_text_encoder: Arc<Mutex<Option<Qwen3Model<Backend>>>>,
    image_transformer: Arc<Mutex<Option<ZImageModel<Backend>>>>,
    image_autoencoder: Arc<Mutex<Option<AutoEncoder<Backend>>>>,

    // Text models
    text_model: Arc<Mutex<Option<Qwen3ForCausalLM<Backend>>>>,
    text_tokenizer: Arc<Mutex<Option<Qwen3Tokenizer>>>,
}

struct ChatMessage {
    role: String,
    content: String,
}

impl ZImageApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let (tx, rx) = mpsc::channel();

        // Default directories
        let project_dirs = directories::ProjectDirs::from("com", "zimage", "ZImageStudio");
        let models_base_dir = project_dirs
            .as_ref()
            .map(|dirs| dirs.data_dir().join("models").to_string_lossy().to_string())
            .unwrap_or_else(|| "./models".to_string());
        let history_dir = project_dirs
            .as_ref()
            .map(|dirs| dirs.data_dir().join("history"))
            .unwrap_or_else(|| PathBuf::from("./history"));

        // Create history directory
        let _ = std::fs::create_dir_all(&history_dir);

        let mut app = Self {
            current_tab: Tab::Image,
            image_model_dir: String::new(),
            text_model_dir: String::new(),
            image_prompt: "A beautiful sunset over mountains".to_string(),
            image_width: 512,
            image_height: 512,
            generated_image: None,
            is_generating_image: false,
            image_models_loaded: false,
            last_generated_path: None,

            // Generation settings
            num_inference_steps: 8,
            use_seed: false,
            seed: 42,

            // Memory settings
            attention_slice_size: 0,
            low_memory_mode: false,

            // History
            history: Vec::new(),
            selected_history_index: None,
            history_dir,

            // Text chat
            chat_messages: Vec::new(),
            chat_input: String::new(),
            is_generating_text: false,
            text_model_loaded: false,
            max_tokens: 128,
            temperature: 0.7,

            // Downloads
            models_base_dir,
            download_progress: std::collections::HashMap::new(),
            download_status: std::collections::HashMap::new(),

            // Logs
            logs: Vec::new(),
            show_logs: true,
            device_initialized: false,

            // Async
            tx,
            rx,
            device: None,

            // Image models
            image_tokenizer: Arc::new(Mutex::new(None)),
            image_text_encoder: Arc::new(Mutex::new(None)),
            image_transformer: Arc::new(Mutex::new(None)),
            image_autoencoder: Arc::new(Mutex::new(None)),

            // Text models
            text_model: Arc::new(Mutex::new(None)),
            text_tokenizer: Arc::new(Mutex::new(None)),
        };

        // Initialize device
        app.initialize_device();

        // Load history
        app.load_history();

        app
    }

    fn load_history(&mut self) {
        // Load history from disk
        if let Ok(entries) = std::fs::read_dir(&self.history_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map(|e| e == "png").unwrap_or(false) {
                    let json_path = path.with_extension("json");
                    if let Ok(json_content) = std::fs::read_to_string(&json_path) {
                        if let Ok(item) = serde_json::from_str::<serde_json::Value>(&json_content) {
                            self.history.push(HistoryItem {
                                prompt: item["prompt"].as_str().unwrap_or("").to_string(),
                                width: item["width"].as_i64().unwrap_or(512) as i32,
                                height: item["height"].as_i64().unwrap_or(512) as i32,
                                steps: item["steps"].as_i64().unwrap_or(8) as i32,
                                timestamp: item["timestamp"].as_str().unwrap_or("").to_string(),
                                image_path: path.clone(),
                                generation_time: item["generation_time"].as_f64().unwrap_or(0.0) as f32,
                            });
                        }
                    }
                }
            }
        }
        self.history.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
    }

    fn save_to_history(&mut self, prompt: &str, width: i32, height: i32, steps: i32, gen_time: f32, image_path: &PathBuf) {
        let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
        let filename = chrono::Local::now().format("%Y%m%d_%H%M%S").to_string();

        let history_image_path = self.history_dir.join(format!("{}.png", filename));
        if std::fs::copy(image_path, &history_image_path).is_ok() {
            let json_path = history_image_path.with_extension("json");
            let metadata = serde_json::json!({
                "prompt": prompt,
                "width": width,
                "height": height,
                "steps": steps,
                "timestamp": timestamp,
                "generation_time": gen_time,
            });
            let _ = std::fs::write(&json_path, serde_json::to_string_pretty(&metadata).unwrap_or_default());

            self.history.insert(0, HistoryItem {
                prompt: prompt.to_string(),
                width,
                height,
                steps,
                timestamp,
                image_path: history_image_path,
                generation_time: gen_time,
            });
        }
    }

    fn initialize_device(&mut self) {
        self.log(&format!("Initializing {} backend...", BACKEND_NAME));
        let device = backend::create_device();
        self.device = Some(device);
        self.device_initialized = true;
        self.log(&format!("{} initialized", BACKEND_NAME));
    }

    fn log(&mut self, msg: &str) {
        let timestamp = chrono::Local::now().format("%H:%M:%S").to_string();
        self.logs.push(format!("[{}] {}", timestamp, msg));
        if self.logs.len() > 200 {
            self.logs.remove(0);
        }
    }

    fn select_directory(&mut self, model_type: ModelType) {
        if let Some(path) = rfd::FileDialog::new()
            .set_title(match model_type {
                ModelType::Image => "Select Image Model Directory",
                ModelType::Text => "Select Text Model Directory",
            })
            .pick_folder()
        {
            let path_str = path.to_string_lossy().to_string();
            match model_type {
                ModelType::Image => self.image_model_dir = path_str,
                ModelType::Text => self.text_model_dir = path_str,
            }
        }
    }

    fn load_text_model(&mut self) {
        if self.text_model_dir.is_empty() {
            self.log("Error: Text model directory not set");
            return;
        }

        let model_dir = PathBuf::from(&self.text_model_dir);
        let device = self.device.clone().unwrap();
        let tx = self.tx.clone();
        let text_model = self.text_model.clone();
        let text_tokenizer = self.text_tokenizer.clone();

        self.log(&format!("Loading Qwen3-0.6B from {:?}...", model_dir));

        thread::spawn(move || {
            let tokenizer_path = model_dir.join("tokenizer.json");
            let tokenizer = match Qwen3Tokenizer::from_file(&tokenizer_path) {
                Ok(t) => t,
                Err(e) => {
                    let _ = tx.send(WorkerMessage::ModelLoadComplete(
                        ModelType::Text,
                        Err(format!("Failed to load tokenizer: {}", e)),
                    ));
                    return;
                }
            };

            let _ = tx.send(WorkerMessage::Log("Tokenizer loaded".to_string()));

            // Load model - prefer .bpk
            let bpk_path = model_dir.join("model.bpk");
            let safetensors_path = model_dir.join("model.safetensors");
            let model_path = if bpk_path.exists() {
                bpk_path
            } else {
                safetensors_path
            };

            let _ = tx.send(WorkerMessage::Log(format!(
                "Loading model from {:?}...",
                model_path
            )));

            let mut model: Qwen3ForCausalLM<Backend> =
                Qwen3Config::qwen3_0_6b().init_causal_lm(&device);

            if let Err(e) = model.load_weights(&model_path) {
                let _ = tx.send(WorkerMessage::ModelLoadComplete(
                    ModelType::Text,
                    Err(format!("Failed to load model: {:?}", e)),
                ));
                return;
            }

            *text_model.lock().unwrap() = Some(model);
            *text_tokenizer.lock().unwrap() = Some(tokenizer);

            let _ = tx.send(WorkerMessage::ModelLoadComplete(ModelType::Text, Ok(())));
        });
    }

    fn generate_text(&mut self) {
        if self.chat_input.trim().is_empty() {
            return;
        }

        let user_text = self.chat_input.trim().to_string();
        self.chat_messages.push(ChatMessage {
            role: "user".to_string(),
            content: user_text.clone(),
        });
        self.chat_input.clear();
        self.is_generating_text = true;

        let tx = self.tx.clone();
        let text_model = self.text_model.clone();
        let text_tokenizer = self.text_tokenizer.clone();
        let device = self.device.clone().unwrap();
        let max_tokens = self.max_tokens;
        let temperature = self.temperature;

        self.log(&format!("Generating response for: {}", user_text));

        thread::spawn(move || {
            let tokenizer_guard = text_tokenizer.lock().unwrap();
            let tokenizer = match tokenizer_guard.as_ref() {
                Some(t) => t,
                None => {
                    let _ = tx.send(WorkerMessage::TextGenComplete(Err(
                        "Tokenizer not loaded".to_string(),
                    )));
                    return;
                }
            };

            let model_guard = text_model.lock().unwrap();
            let model = match model_guard.as_ref() {
                Some(m) => m,
                None => {
                    let _ = tx.send(WorkerMessage::TextGenComplete(Err(
                        "Model not loaded".to_string(),
                    )));
                    return;
                }
            };

            let formatted = tokenizer.apply_chat_template(&user_text);

            let (input_ids, _) = match tokenizer.encode_no_pad(&formatted) {
                Ok(r) => r,
                Err(e) => {
                    let _ = tx.send(WorkerMessage::TextGenComplete(Err(format!(
                        "Tokenization failed: {}",
                        e
                    ))));
                    return;
                }
            };

            let _ = tx.send(WorkerMessage::Log(format!(
                "Input: {} tokens",
                input_ids.len()
            )));

            let input_ids_i64: Vec<i64> = input_ids.iter().map(|&x| x as i64).collect();
            let input_tensor: Tensor<Backend, 2, Int> =
                Tensor::<Backend, 1, Int>::from_data(input_ids_i64.as_slice(), &device)
                    .reshape([1, input_ids.len()]);

            let output = model.generate_with_cache(
                input_tensor,
                max_tokens as usize,
                temperature,
                0.9,
                50,
            );

            let output_data = output.into_data();
            let output_ids: Vec<u32> = output_data
                .as_slice::<i64>()
                .unwrap_or(&[])
                .iter()
                .skip(input_ids.len())
                .map(|&x| x as u32)
                .collect();

            let text = tokenizer.decode(&output_ids).unwrap_or_default();

            let mut clean_text = text;
            if let Some(pos) = clean_text.find("<|im_end|>") {
                clean_text = clean_text[..pos].to_string();
            }
            if let Some(pos) = clean_text.find("<|endoftext|>") {
                clean_text = clean_text[..pos].to_string();
            }

            let _ = tx.send(WorkerMessage::TextGenComplete(Ok(clean_text.trim().to_string())));
        });
    }

    fn load_image_models(&mut self) {
        if self.image_model_dir.is_empty() {
            self.log("Error: Image model directory not set");
            return;
        }

        let model_dir = PathBuf::from(&self.image_model_dir);
        let device = self.device.clone().unwrap();
        let tx = self.tx.clone();

        let image_tokenizer = self.image_tokenizer.clone();
        let image_text_encoder = self.image_text_encoder.clone();
        let image_transformer = self.image_transformer.clone();
        let image_autoencoder = self.image_autoencoder.clone();

        self.log(&format!("Loading image models from {:?}...", model_dir));

        thread::spawn(move || {
            let start = std::time::Instant::now();

            let tokenizer_path = model_dir.join("qwen3-tokenizer.json");
            let _ = tx.send(WorkerMessage::Log("Loading tokenizer...".to_string()));
            let tokenizer = match Qwen3Tokenizer::from_file(&tokenizer_path) {
                Ok(t) => t,
                Err(e) => {
                    let _ = tx.send(WorkerMessage::ModelLoadComplete(
                        ModelType::Image,
                        Err(format!("Failed to load tokenizer: {}", e)),
                    ));
                    return;
                }
            };
            *image_tokenizer.lock().unwrap() = Some(tokenizer);
            let _ = tx.send(WorkerMessage::Log("Tokenizer loaded".to_string()));

            let te_bpk = model_dir.join("qwen3_4b_text_encoder.bpk");
            let te_safetensors = model_dir.join("qwen3_4b_text_encoder.safetensors");
            let te_path = if te_bpk.exists() { te_bpk } else { te_safetensors };
            let _ = tx.send(WorkerMessage::Log(format!("Loading text encoder from {:?}...", te_path)));

            let mut text_encoder: Qwen3Model<Backend> = Qwen3Config::z_image_text_encoder().init(&device);
            if let Err(e) = text_encoder.load_weights(&te_path) {
                let _ = tx.send(WorkerMessage::ModelLoadComplete(
                    ModelType::Image,
                    Err(format!("Failed to load text encoder: {:?}", e)),
                ));
                return;
            }
            *image_text_encoder.lock().unwrap() = Some(text_encoder);
            let _ = tx.send(WorkerMessage::Log("Text encoder loaded".to_string()));

            let transformer_path = model_dir.join("z_image_turbo_bf16.bpk");
            let _ = tx.send(WorkerMessage::Log("Loading transformer...".to_string()));

            let mut transformer: ZImageModel<Backend> = ZImageModelConfig::default().init(&device);
            if let Err(e) = transformer.load_weights(&transformer_path) {
                let _ = tx.send(WorkerMessage::ModelLoadComplete(
                    ModelType::Image,
                    Err(format!("Failed to load transformer: {:?}", e)),
                ));
                return;
            }
            *image_transformer.lock().unwrap() = Some(transformer);
            let _ = tx.send(WorkerMessage::Log("Transformer loaded".to_string()));

            let ae_bpk = model_dir.join("ae.bpk");
            let ae_safetensors = model_dir.join("ae.safetensors");
            let ae_path = if ae_bpk.exists() { ae_bpk } else { ae_safetensors };
            let _ = tx.send(WorkerMessage::Log(format!("Loading autoencoder from {:?}...", ae_path)));

            let mut ae: AutoEncoder<Backend> = AutoEncoderConfig::flux_ae().init(&device);
            if let Err(e) = ae.load_weights(&ae_path) {
                let _ = tx.send(WorkerMessage::ModelLoadComplete(
                    ModelType::Image,
                    Err(format!("Failed to load autoencoder: {:?}", e)),
                ));
                return;
            }
            *image_autoencoder.lock().unwrap() = Some(ae);

            let elapsed = start.elapsed().as_secs_f32();
            let _ = tx.send(WorkerMessage::Log(format!("All image models loaded in {:.1}s", elapsed)));
            let _ = tx.send(WorkerMessage::ModelLoadComplete(ModelType::Image, Ok(())));
        });
    }

    fn unload_image_models(&mut self) {
        *self.image_tokenizer.lock().unwrap() = None;
        *self.image_text_encoder.lock().unwrap() = None;
        *self.image_transformer.lock().unwrap() = None;
        *self.image_autoencoder.lock().unwrap() = None;
        self.image_models_loaded = false;
        self.log("Image models unloaded");
    }

    fn generate_image(&mut self) {
        if self.image_prompt.trim().is_empty() {
            self.log("Error: Prompt is empty");
            return;
        }

        if !self.image_models_loaded {
            self.log("Error: Image models not loaded. Please load models first.");
            return;
        }

        self.is_generating_image = true;
        let width = self.image_width;
        let height = self.image_height;
        let steps = self.num_inference_steps;
        let prompt = self.image_prompt.clone();

        self.log(&format!(
            "Generating {}x{} image with {} steps...",
            width, height, steps
        ));

        z_image::set_attention_slice_size(self.attention_slice_size as usize);

        let tx = self.tx.clone();
        let device = self.device.clone().unwrap();
        let image_tokenizer = self.image_tokenizer.clone();
        let image_text_encoder = self.image_text_encoder.clone();
        let image_transformer = self.image_transformer.clone();
        let image_autoencoder = self.image_autoencoder.clone();
        let seed = if self.use_seed { Some(self.seed) } else { None };

        let output_path = std::env::temp_dir().join(format!("z_image_{}.png", chrono::Local::now().format("%Y%m%d_%H%M%S")));

        thread::spawn(move || {
            let _start = std::time::Instant::now();

            let tokenizer_guard = image_tokenizer.lock().unwrap();
            let tokenizer = match tokenizer_guard.as_ref() {
                Some(t) => t,
                None => {
                    let _ = tx.send(WorkerMessage::ImageGenComplete(Err("Tokenizer not loaded".to_string())));
                    return;
                }
            };

            let text_encoder_guard = image_text_encoder.lock().unwrap();
            let text_encoder = match text_encoder_guard.as_ref() {
                Some(t) => t,
                None => {
                    let _ = tx.send(WorkerMessage::ImageGenComplete(Err("Text encoder not loaded".to_string())));
                    return;
                }
            };

            let transformer_guard = image_transformer.lock().unwrap();
            let transformer = match transformer_guard.as_ref() {
                Some(t) => t,
                None => {
                    let _ = tx.send(WorkerMessage::ImageGenComplete(Err("Transformer not loaded".to_string())));
                    return;
                }
            };

            let autoencoder_guard = image_autoencoder.lock().unwrap();
            let autoencoder = match autoencoder_guard.as_ref() {
                Some(a) => a,
                None => {
                    let _ = tx.send(WorkerMessage::ImageGenComplete(Err("Autoencoder not loaded".to_string())));
                    return;
                }
            };

            let opts = z_image::GenerateFromTextOpts {
                prompt: prompt.clone(),
                out_path: output_path.clone(),
                width: width as usize,
                height: height as usize,
                num_inference_steps: Some(steps as usize),
                seed,
            };

            match z_image::generate_from_text(&opts, tokenizer, text_encoder, autoencoder, transformer, &device) {
                Ok(()) => {
                    let elapsed = _start.elapsed().as_secs_f32();
                    let _ = tx.send(WorkerMessage::Log(format!("Generation complete in {:.1}s", elapsed)));
                    let _ = tx.send(WorkerMessage::ImageGenComplete(Ok(output_path)));
                }
                Err(e) => {
                    let _ = tx.send(WorkerMessage::ImageGenComplete(Err(format!("Generation failed: {:?}", e))));
                }
            }
        });
    }

    fn process_messages(&mut self, ctx: &egui::Context) {
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                WorkerMessage::Log(text) => self.log(&text),
                WorkerMessage::ImageGenComplete(result) => {
                    self.is_generating_image = false;
                    match result {
                        Ok(path) => {
                            self.log(&format!("Image saved to {:?}", path));
                            self.last_generated_path = Some(path.clone());

                            let prompt = self.image_prompt.clone();
                            let width = self.image_width;
                            let height = self.image_height;
                            let steps = self.num_inference_steps;
                            self.save_to_history(
                                &prompt,
                                width,
                                height,
                                steps,
                                0.0,
                                &path,
                            );

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
                                        "generated_image",
                                        color_image,
                                        egui::TextureOptions::default(),
                                    ));
                                }
                            }
                        }
                        Err(e) => self.log(&format!("Image generation failed: {}", e)),
                    }
                }
                WorkerMessage::TextGenComplete(result) => {
                    self.is_generating_text = false;
                    match result {
                        Ok(text) => {
                            self.chat_messages.push(ChatMessage {
                                role: "assistant".to_string(),
                                content: text,
                            });
                            self.log("Text generation complete");
                        }
                        Err(e) => {
                            self.log(&format!("Text generation failed: {}", e));
                            self.chat_messages.push(ChatMessage {
                                role: "assistant".to_string(),
                                content: format!("[Error: {}]", e),
                            });
                        }
                    }
                }
                WorkerMessage::ModelLoadComplete(model_type, result) => match model_type {
                    ModelType::Text => match result {
                        Ok(()) => {
                            self.text_model_loaded = true;
                            self.log("Text model loaded successfully");
                        }
                        Err(e) => self.log(&format!("Failed to load text model: {}", e)),
                    },
                    ModelType::Image => match result {
                        Ok(()) => {
                            self.image_models_loaded = true;
                            self.log("Image models loaded successfully");
                        }
                        Err(e) => self.log(&format!("Failed to load image models: {}", e)),
                    },
                },
                WorkerMessage::DownloadProgress(id, progress) => {
                    self.download_progress.insert(id.clone(), progress);
                    self.download_status
                        .insert(id, format!("{}%", (progress * 100.0) as i32));
                }
                WorkerMessage::DownloadComplete(id, result) => {
                    self.download_progress.insert(id.clone(), 1.0);
                    match result {
                        Ok(()) => {
                            self.download_status.insert(id, "Complete".to_string());
                        }
                        Err(e) => {
                            self.download_status.insert(id, format!("Error: {}", e));
                        }
                    }
                }
            }
        }
    }
}

impl eframe::App for ZImageApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.process_messages(ctx);

        egui::TopBottomPanel::top("header").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Z-Image Studio");
                ui.separator();

                ui.selectable_value(&mut self.current_tab, Tab::Image, "Image");
                ui.selectable_value(&mut self.current_tab, Tab::History, "History");
                ui.selectable_value(&mut self.current_tab, Tab::Chat, "Chat");
                ui.selectable_value(&mut self.current_tab, Tab::Settings, "Settings");
                ui.selectable_value(&mut self.current_tab, Tab::Models, "Models");

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .selectable_label(self.show_logs, "Logs")
                        .clicked()
                    {
                        self.show_logs = !self.show_logs;
                    }

                    let (color, text) = if self.device_initialized {
                        (egui::Color32::GREEN, "GPU Ready")
                    } else {
                        (egui::Color32::RED, "Initializing...")
                    };
                    ui.colored_label(color, text);
                });
            });
        });

        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if self.device_initialized {
                    ui.label(format!("Backend: {}", BACKEND_NAME));
                } else {
                    ui.spinner();
                    ui.label("Initializing...");
                }
            });
        });

        if self.show_logs {
            egui::SidePanel::right("logs")
                .min_width(300.0)
                .max_width(400.0)
                .show(ctx, |ui| {
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
                                let color = if log.contains("Error") || log.contains("ERROR") {
                                    egui::Color32::RED
                                } else if log.contains("Warning") {
                                    egui::Color32::YELLOW
                                } else {
                                    egui::Color32::GRAY
                                };
                                ui.colored_label(color, log);
                            }
                        });
                });
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            match self.current_tab {
                Tab::Image => self.show_image_tab(ui, ctx),
                Tab::History => self.show_history_tab(ui, ctx),
                Tab::Chat => self.show_chat_tab(ui),
                Tab::Settings => self.show_settings_tab(ui),
                Tab::Models => self.show_models_tab(ui),
            }
        });

        if self.is_generating_image || self.is_generating_text {
            ctx.request_repaint();
        }
    }
}

impl ZImageApp {
    fn show_image_tab(&mut self, ui: &mut egui::Ui, _ctx: &egui::Context) {
        ui.heading("Image Generation");
        ui.separator();

        ui.horizontal(|ui| {
            ui.label("Model Directory:");
            ui.add(
                egui::TextEdit::singleline(&mut self.image_model_dir)
                    .desired_width(300.0)
                    .hint_text("Select model directory..."),
            );
            if ui.button("Browse...").clicked() {
                self.select_directory(ModelType::Image);
            }

            if !self.image_model_dir.is_empty() {
                if self.image_models_loaded {
                    ui.colored_label(egui::Color32::GREEN, "Models loaded");
                    if ui.button("Unload").clicked() {
                        self.unload_image_models();
                    }
                } else {
                    if ui.button("Load Models").clicked() {
                        self.load_image_models();
                    }
                }
            }
        });

        ui.add_space(10.0);

        ui.label("Prompt:");
        ui.add(
            egui::TextEdit::multiline(&mut self.image_prompt)
                .desired_rows(3)
                .desired_width(f32::INFINITY),
        );

        ui.add_space(10.0);

        ui.horizontal(|ui| {
            ui.label("Width:");
            ui.add(egui::DragValue::new(&mut self.image_width).range(256..=1024).speed(16));

            ui.label("Height:");
            ui.add(egui::DragValue::new(&mut self.image_height).range(256..=1024).speed(16));

            ui.separator();

            ui.label("Steps:");
            ui.add(egui::Slider::new(&mut self.num_inference_steps, 4..=20));
        });

        ui.horizontal(|ui| {
            ui.label("Presets:");
            if ui.button("256x256").clicked() {
                self.image_width = 256;
                self.image_height = 256;
            }
            if ui.button("512x512").clicked() {
                self.image_width = 512;
                self.image_height = 512;
            }
            if ui.button("768x768").clicked() {
                self.image_width = 768;
                self.image_height = 768;
            }
            if ui.button("1024x1024").clicked() {
                self.image_width = 1024;
                self.image_height = 1024;
            }
        });

        ui.add_space(5.0);

        ui.horizontal(|ui| {
            ui.checkbox(&mut self.use_seed, "Use seed:");
            ui.add_enabled(
                self.use_seed,
                egui::DragValue::new(&mut self.seed).speed(1),
            );
            if ui.button("Random").clicked() {
                self.seed = rand::random();
            }
        });

        ui.add_space(10.0);

        ui.horizontal(|ui| {
            let can_generate = !self.is_generating_image && self.image_models_loaded;

            if ui
                .add_enabled(can_generate, egui::Button::new("Generate"))
                .clicked()
            {
                self.generate_image();
            }

            if self.is_generating_image {
                ui.spinner();
                ui.label("Generating...");
            }

            if !self.image_models_loaded && !self.image_model_dir.is_empty() {
                ui.colored_label(egui::Color32::YELLOW, "Please load models first");
            }
        });

        ui.add_space(20.0);

        let available_size = ui.available_size();
        let image_size = available_size.min_elem().min(512.0);

        egui::Frame::canvas(ui.style()).show(ui, |ui| {
            ui.set_min_size(egui::vec2(image_size, image_size));

            if let Some(texture) = &self.generated_image {
                ui.image(texture);
            } else {
                ui.centered_and_justified(|ui| {
                    ui.label("Generated image will appear here");
                });
            }
        });
    }

    fn show_history_tab(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.heading("Generation History");
        ui.separator();

        if self.history.is_empty() {
            ui.label("No images generated yet.");
            return;
        }

        ui.horizontal(|ui| {
            egui::ScrollArea::vertical()
                .max_width(200.0)
                .show(ui, |ui| {
                    for (i, item) in self.history.iter().enumerate() {
                        let selected = self.selected_history_index == Some(i);
                        if ui.selectable_label(selected, &item.timestamp).clicked() {
                            self.selected_history_index = Some(i);
                        }
                        ui.label(format!("{}x{}", item.width, item.height));
                        ui.add_space(5.0);
                    }
                });

            ui.separator();

            if let Some(idx) = self.selected_history_index {
                let item_data = self.history.get(idx).cloned();

                if let Some(item) = item_data {
                    ui.vertical(|ui| {
                        ui.heading("Details");
                        ui.label(format!("Prompt: {}", item.prompt));
                        ui.label(format!("Size: {}x{}", item.width, item.height));
                        ui.label(format!("Steps: {}", item.steps));

                        ui.add_space(10.0);

                        if let Ok(image_data) = std::fs::read(&item.image_path) {
                            if let Ok(image) = image::load_from_memory(&image_data) {
                                let size = [image.width() as _, image.height() as _];
                                let image_buffer = image.to_rgba8();
                                let pixels = image_buffer.as_flat_samples();
                                let color_image = egui::ColorImage::from_rgba_unmultiplied(
                                    size,
                                    pixels.as_slice(),
                                );
                                let texture = ctx.load_texture(
                                    format!("history_{}", idx),
                                    color_image,
                                    egui::TextureOptions::default(),
                                );
                                ui.image(&texture);
                            }
                        }

                        ui.add_space(10.0);

                        let mut should_delete = false;
                        let mut should_use_prompt = false;
                        let mut should_open = false;

                        ui.horizontal(|ui| {
                            should_use_prompt = ui.button("Use Prompt").clicked();
                            should_open = ui.button("Open File").clicked();
                            should_delete = ui.button("Delete").clicked();
                        });

                        if should_use_prompt {
                            self.image_prompt = item.prompt.clone();
                            self.image_width = item.width;
                            self.image_height = item.height;
                            self.num_inference_steps = item.steps;
                            self.current_tab = Tab::Image;
                        }

                        if should_open {
                            let _ = open::that(&item.image_path);
                        }

                        if should_delete {
                            let _ = std::fs::remove_file(&item.image_path);
                            let json_path = item.image_path.with_extension("json");
                            let _ = std::fs::remove_file(&json_path);
                            self.history.remove(idx);
                            self.selected_history_index = None;
                        }
                    });
                }
            } else {
                ui.label("Select an image from the list");
            }
        });
    }

    fn show_settings_tab(&mut self, ui: &mut egui::Ui) {
        ui.heading("Settings");
        ui.separator();

        ui.group(|ui| {
            ui.heading("Generation Settings");

            ui.horizontal(|ui| {
                ui.label("Inference Steps:");
                ui.add(egui::Slider::new(&mut self.num_inference_steps, 4..=20));
            });
            ui.label("More steps = higher quality but slower.");

            ui.add_space(10.0);

            ui.horizontal(|ui| {
                ui.checkbox(&mut self.use_seed, "Use Fixed Seed");
                if self.use_seed {
                    ui.add(egui::DragValue::new(&mut self.seed).speed(1));
                    if ui.button("Randomize").clicked() {
                        self.seed = rand::random();
                    }
                }
            });
        });

        ui.add_space(20.0);

        ui.group(|ui| {
            ui.heading("Memory Optimization");

            ui.horizontal(|ui| {
                ui.label("Attention Slice Size:");
                ui.add(egui::Slider::new(&mut self.attention_slice_size, 0..=8));
            });
            ui.label("0 = No slicing (fastest). Higher = less memory but slower.");

            ui.add_space(10.0);

            ui.checkbox(&mut self.low_memory_mode, "Low Memory Mode");
            ui.label("Unloads text encoder during diffusion to save ~7.5GB.");
        });

        ui.add_space(20.0);

        ui.group(|ui| {
            ui.heading("Compute Backend");

            ui.horizontal(|ui| {
                ui.label("Active Backend:");
                ui.colored_label(egui::Color32::GREEN, BACKEND_NAME);
            });

            ui.add_space(10.0);
            ui.label("The backend is selected at compile time.");

            egui::CollapsingHeader::new("Build Commands").show(ui, |ui| {
                ui.label("macOS (Metal):");
                ui.code("cargo build --release --features \"egui,metal\"");
                ui.add_space(5.0);

                ui.label("Windows/Linux (NVIDIA CUDA):");
                ui.code("cargo build --release --features \"egui,cuda\"");
                ui.add_space(5.0);

                ui.label("Windows/Linux (Vulkan):");
                ui.code("cargo build --release --features \"egui,vulkan\"");
                ui.add_space(5.0);

                ui.label("CPU only:");
                ui.code("cargo build --release --features \"egui,cpu\"");
            });
        });
    }

    fn show_chat_tab(&mut self, ui: &mut egui::Ui) {
        ui.heading("Qwen3 Chat (0.6B)");
        ui.separator();

        ui.horizontal(|ui| {
            ui.label("Model Directory:");
            ui.add(
                egui::TextEdit::singleline(&mut self.text_model_dir)
                    .desired_width(300.0)
                    .hint_text("Select model directory..."),
            );
            if ui.button("Browse...").clicked() {
                self.select_directory(ModelType::Text);
            }

            if !self.text_model_dir.is_empty() && !self.text_model_loaded {
                if ui.button("Load Model").clicked() {
                    self.load_text_model();
                }
            }

            if self.text_model_loaded {
                ui.colored_label(egui::Color32::GREEN, "Model loaded");
            }
        });

        ui.add_space(10.0);

        ui.horizontal(|ui| {
            ui.label("Max tokens:");
            ui.add(egui::Slider::new(&mut self.max_tokens, 32..=512));

            ui.separator();

            ui.label("Temperature:");
            ui.add(egui::Slider::new(&mut self.temperature, 0.0..=1.5));
        });

        ui.separator();

        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .stick_to_bottom(true)
            .max_height(ui.available_height() - 60.0)
            .show(ui, |ui| {
                for msg in &self.chat_messages {
                    let (align, color) = if msg.role == "user" {
                        (egui::Align::RIGHT, egui::Color32::from_rgb(59, 130, 246))
                    } else {
                        (egui::Align::LEFT, egui::Color32::from_rgb(100, 100, 100))
                    };

                    ui.with_layout(egui::Layout::top_down(align), |ui| {
                        ui.label(if msg.role == "user" { "You" } else { "Qwen3" });

                        egui::Frame::none()
                            .fill(color)
                            .rounding(8.0)
                            .inner_margin(12.0)
                            .show(ui, |ui| {
                                ui.colored_label(egui::Color32::WHITE, &msg.content);
                            });
                    });

                    ui.add_space(8.0);
                }

                if self.is_generating_text {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label("Thinking...");
                    });
                }
            });

        ui.separator();
        ui.horizontal(|ui| {
            let response = ui.add(
                egui::TextEdit::singleline(&mut self.chat_input)
                    .desired_width(ui.available_width() - 80.0)
                    .hint_text("Type a message...")
                    .interactive(self.text_model_loaded && !self.is_generating_text),
            );

            let can_send = self.text_model_loaded
                && !self.is_generating_text
                && !self.chat_input.trim().is_empty();

            if ui.add_enabled(can_send, egui::Button::new("Send")).clicked()
                || (response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) && can_send)
            {
                self.generate_text();
            }
        });
    }

    fn show_models_tab(&mut self, ui: &mut egui::Ui) {
        ui.heading("Model Management");
        ui.separator();

        ui.label("Download and configure AI models.");
        ui.add_space(10.0);

        ui.group(|ui| {
            ui.label("Download Location:");
            ui.horizontal(|ui| {
                ui.add(
                    egui::TextEdit::singleline(&mut self.models_base_dir)
                        .desired_width(400.0),
                );
                if ui.button("Browse...").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .set_title("Select Models Directory")
                        .pick_folder()
                    {
                        self.models_base_dir = path.to_string_lossy().to_string();
                    }
                }
            });
        });

        ui.add_space(20.0);

        ui.group(|ui| {
            ui.heading("Image Generation Models (~20 GB)");
            ui.label("From holgt/z-image-burn");
            ui.add_space(10.0);

            self.model_download_row(
                ui,
                "Z-Image Turbo Transformer",
                "z_image_turbo_bf16.bpk",
                "12.3 GB",
                "https://huggingface.co/holgt/z-image-burn/resolve/main/z_image_turbo_bf16.bpk",
                "z-image",
            );

            self.model_download_row(
                ui,
                "Qwen3-4B Text Encoder",
                "qwen3_4b_text_encoder.bpk",
                "8.0 GB",
                "https://huggingface.co/holgt/z-image-burn/resolve/main/qwen3_4b_text_encoder.bpk",
                "z-image",
            );

            self.model_download_row(
                ui,
                "Flux Autoencoder",
                "ae.bpk",
                "198 MB",
                "https://huggingface.co/holgt/z-image-burn/resolve/main/ae.bpk",
                "z-image",
            );

            self.model_download_row(
                ui,
                "Qwen3 Tokenizer",
                "qwen3-tokenizer.json",
                "11 MB",
                "https://huggingface.co/holgt/z-image-burn/resolve/main/qwen3-tokenizer.json",
                "z-image",
            );

            ui.add_space(10.0);
            if ui.button("Set as Image Model Directory").clicked() {
                self.image_model_dir = format!("{}/z-image", self.models_base_dir);
                self.log("Image model directory set");
            }
        });

        ui.add_space(20.0);

        ui.group(|ui| {
            ui.heading("Text Chat Model (~1.5 GB)");
            ui.label("From holgt/qwen3-0.6b-burn");
            ui.add_space(10.0);

            self.model_download_row(
                ui,
                "Qwen3-0.6B Weights",
                "model.bpk",
                "1.5 GB",
                "https://huggingface.co/holgt/qwen3-0.6b-burn/resolve/main/model.bpk",
                "qwen3-0.6b",
            );

            self.model_download_row(
                ui,
                "Qwen3-0.6B Tokenizer",
                "tokenizer.json",
                "11 MB",
                "https://huggingface.co/holgt/qwen3-0.6b-burn/resolve/main/tokenizer.json",
                "qwen3-0.6b",
            );

            ui.add_space(10.0);
            if ui.button("Set as Text Model Directory").clicked() {
                self.text_model_dir = format!("{}/qwen3-0.6b", self.models_base_dir);
                self.log("Text model directory set");
            }
        });

        ui.add_space(20.0);

        ui.group(|ui| {
            ui.heading("Manual Download");
            ui.hyperlink_to(
                "Z-Image Burn Models",
                "https://huggingface.co/holgt/z-image-burn",
            );
            ui.hyperlink_to(
                "Qwen3-0.6B Burn",
                "https://huggingface.co/holgt/qwen3-0.6b-burn",
            );
        });
    }

    fn model_download_row(
        &mut self,
        ui: &mut egui::Ui,
        name: &str,
        filename: &str,
        size: &str,
        url: &str,
        subdir: &str,
    ) {
        let id = filename.replace('.', "_");
        let dest_dir = format!("{}/{}", self.models_base_dir, subdir);
        let file_path = format!("{}/{}", dest_dir, filename);
        let file_exists = std::path::Path::new(&file_path).exists();

        ui.horizontal(|ui| {
            ui.label(format!("{} ({})", name, size));

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if file_exists {
                    ui.colored_label(egui::Color32::GREEN, "Ready");
                } else if let Some(progress) = self.download_progress.get(&id) {
                    ui.add(egui::ProgressBar::new(*progress).show_percentage());
                } else if ui.button("Download").clicked() {
                    self.start_download(&id, url, &dest_dir);
                }
            });
        });
    }

    fn start_download(&mut self, id: &str, url: &str, dest_dir: &str) {
        let id = id.to_string();
        let url = url.to_string();
        let dest_dir = dest_dir.to_string();
        let tx = self.tx.clone();

        self.download_progress.insert(id.clone(), 0.0);
        self.log(&format!("Starting download: {}", url));

        thread::spawn(move || {
            if let Err(e) = std::fs::create_dir_all(&dest_dir) {
                let _ = tx.send(WorkerMessage::DownloadComplete(
                    id,
                    Err(format!("Failed to create directory: {}", e)),
                ));
                return;
            }

            let filename = url.split('/').last().unwrap_or("download");
            let file_path = format!("{}/{}", dest_dir, filename);

            match reqwest::blocking::get(&url) {
                Ok(response) => {
                    let total_size = response.content_length().unwrap_or(0);
                    let mut file = match std::fs::File::create(&file_path) {
                        Ok(f) => f,
                        Err(e) => {
                            let _ = tx.send(WorkerMessage::DownloadComplete(
                                id,
                                Err(format!("Failed to create file: {}", e)),
                            ));
                            return;
                        }
                    };

                    let mut downloaded: u64 = 0;
                    let mut reader = response;

                    loop {
                        use std::io::Read;
                        let mut buffer = [0u8; 8192];
                        match reader.read(&mut buffer) {
                            Ok(0) => break,
                            Ok(n) => {
                                use std::io::Write;
                                if let Err(e) = file.write_all(&buffer[..n]) {
                                    let _ = tx.send(WorkerMessage::DownloadComplete(
                                        id,
                                        Err(format!("Write error: {}", e)),
                                    ));
                                    return;
                                }
                                downloaded += n as u64;
                                if total_size > 0 {
                                    let progress = downloaded as f32 / total_size as f32;
                                    let _ = tx.send(WorkerMessage::DownloadProgress(
                                        id.clone(),
                                        progress,
                                    ));
                                }
                            }
                            Err(e) => {
                                let _ = tx.send(WorkerMessage::DownloadComplete(
                                    id,
                                    Err(format!("Read error: {}", e)),
                                ));
                                return;
                            }
                        }
                    }

                    let _ = tx.send(WorkerMessage::DownloadComplete(id, Ok(())));
                }
                Err(e) => {
                    let _ = tx.send(WorkerMessage::DownloadComplete(
                        id,
                        Err(format!("Download failed: {}", e)),
                    ));
                }
            }
        });
    }
}
