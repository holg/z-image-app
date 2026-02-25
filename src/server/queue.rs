use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Instant;

use burn::tensor::backend::Backend;
use tokio::sync::mpsc;

use super::state::{GenerationJob, TaskInfo, TaskStatus};

/// Starts the GPU worker on a dedicated OS thread.
///
/// The worker creates the device and loads all models on its own thread,
/// then processes generation jobs from the channel sequentially.
pub fn spawn_gpu_worker<B: Backend>(
    mut rx: mpsc::UnboundedReceiver<GenerationJob>,
    tasks: Arc<RwLock<HashMap<String, TaskInfo>>>,
    models_loaded: Arc<AtomicBool>,
    model_dir: PathBuf,
    output_dir: PathBuf,
    create_device: fn() -> B::Device,
) {
    std::thread::spawn(move || {
        let device = create_device();

        eprintln!(
            "[server] GPU worker starting, loading models from {:?}...",
            model_dir
        );
        let total_start = Instant::now();

        // Load tokenizer
        let tokenizer_path = find_tokenizer(&model_dir);
        let tokenizer = match qwen3_burn::Qwen3Tokenizer::from_file(&tokenizer_path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("[server] FATAL: Failed to load tokenizer: {}", e);
                return;
            }
        };
        eprintln!("[server] Tokenizer loaded");

        // Load text encoder
        let te_bpk = model_dir.join("qwen3_4b_text_encoder.bpk");
        let te_safetensors = model_dir.join("qwen3_4b_text_encoder.safetensors");
        let text_encoder_path = if te_bpk.exists() { te_bpk } else { te_safetensors };
        let mut text_encoder: qwen3_burn::Qwen3Model<B> =
            qwen3_burn::Qwen3Config::z_image_text_encoder().init(&device);
        if let Err(e) = text_encoder.load_weights(&text_encoder_path) {
            eprintln!("[server] FATAL: Failed to load text encoder: {:?}", e);
            return;
        }
        eprintln!("[server] Text encoder loaded");

        // Load transformer - prefer f16, then bf16, then safetensors
        let transformer_path = find_model_file(&model_dir, &[
            "z_image_turbo_f16",
            "z_image_turbo_bf16",
            "z_image_turbo",
        ]);
        eprintln!("[server] Loading transformer from {:?}...", transformer_path);
        let mut transformer: z_image::modules::transformer::ZImageModel<B> =
            z_image::modules::transformer::ZImageModelConfig::default().init(&device);
        if let Err(e) = transformer.load_weights(&transformer_path) {
            eprintln!("[server] FATAL: Failed to load transformer: {:?}", e);
            return;
        }
        eprintln!("[server] Transformer loaded");

        // Load autoencoder
        let ae_bpk = model_dir.join("ae.bpk");
        let ae_safetensors = model_dir.join("ae.safetensors");
        let ae_path = if ae_bpk.exists() { ae_bpk } else { ae_safetensors };
        let mut autoencoder: z_image::modules::ae::AutoEncoder<B> =
            z_image::modules::ae::AutoEncoderConfig::flux_ae().init(&device);
        if let Err(e) = autoencoder.load_weights(&ae_path) {
            eprintln!("[server] FATAL: Failed to load autoencoder: {:?}", e);
            return;
        }
        eprintln!("[server] Autoencoder loaded");

        let load_time = total_start.elapsed().as_secs_f32();
        eprintln!("[server] All models loaded in {:.2}s", load_time);
        models_loaded.store(true, Ordering::Release);

        // Create output directory
        if let Err(e) = std::fs::create_dir_all(&output_dir) {
            eprintln!(
                "[server] Warning: Failed to create output directory: {}",
                e
            );
        }

        // Process jobs using a single-threaded tokio runtime for async channel recv
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("Failed to create tokio runtime for GPU worker");

        rt.block_on(async {
            while let Some(job) = rx.recv().await {
                eprintln!(
                    "[server] Processing job {} - prompt: \"{}\"",
                    job.task_id, job.prompt
                );

                // Update status to processing
                {
                    let mut tasks_w = tasks.write().unwrap();
                    if let Some(task) = tasks_w.get_mut(&job.task_id) {
                        task.status = TaskStatus::Processing;
                    }
                }

                let out_path = output_dir.join(format!("{}.png", job.task_id));
                let gen_start = Instant::now();

                let steps = job.steps.unwrap_or(8);
                let opts = z_image::GenerateFromTextOpts {
                    prompt: job.prompt.clone(),
                    out_path: out_path.clone(),
                    width: job.width,
                    height: job.height,
                    num_inference_steps: Some(steps),
                    seed: job.seed,
                };

                let result = z_image::generate_from_text(
                    &opts,
                    &tokenizer,
                    &text_encoder,
                    &autoencoder,
                    &transformer,
                    &device,
                );

                let gen_time = gen_start.elapsed().as_secs_f32();

                // Update task with result
                let mut tasks_w = tasks.write().unwrap();
                if let Some(task) = tasks_w.get_mut(&job.task_id) {
                    match result {
                        Ok(_) => {
                            eprintln!(
                                "[server] Job {} completed in {:.2}s",
                                job.task_id, gen_time
                            );
                            task.status = TaskStatus::Completed;
                            task.output_path = Some(out_path);
                        }
                        Err(e) => {
                            let err_msg = format!("{:?}", e);
                            eprintln!("[server] Job {} failed: {}", job.task_id, err_msg);
                            task.status = TaskStatus::Failed;
                            task.error = Some(err_msg);
                        }
                    }
                }
            }
        });

        eprintln!("[server] GPU worker shutting down");
    });
}

fn find_tokenizer(model_dir: &PathBuf) -> PathBuf {
    let p1 = model_dir.join("qwen3_tokenizer.json");
    let p2 = model_dir.join("qwen3-tokenizer.json");
    if p1.exists() {
        p1
    } else {
        p2
    }
}

/// Find a model file by trying multiple base names with .bpk then .safetensors extensions.
fn find_model_file(model_dir: &PathBuf, base_names: &[&str]) -> PathBuf {
    for name in base_names {
        let bpk = model_dir.join(format!("{}.bpk", name));
        if bpk.exists() {
            return bpk;
        }
        let st = model_dir.join(format!("{}.safetensors", name));
        if st.exists() {
            return st;
        }
    }
    // Fallback to first name with .bpk (will fail with a clear error)
    model_dir.join(format!("{}.bpk", base_names[0]))
}
