//! Low-memory test binary for Z-Image generation
//!
//! This binary uses a sequential loading approach to minimize VRAM usage:
//! 1. Load text encoder (~4GB Q8), compute embedding, then drop it
//! 2. Load transformer (~6GB Q8) + autoencoder (~0.2GB)
//! 3. Generate using cached embedding

use std::path::PathBuf;

#[cfg(feature = "native-cuda")]
use burn::backend::{Cuda, cuda::CudaDevice};
#[cfg(feature = "native-cuda")]
use burn::tensor::backend::Backend as BackendTrait;
#[cfg(feature = "native-cuda")]
use burn::Tensor;

use qwen3_burn::{Qwen3Config, Qwen3Model, Qwen3Tokenizer};
use z_image::{GenerateWithEmbeddingOpts, modules::ae::AutoEncoderConfig, modules::transformer::ZImageModelConfig};

#[cfg(feature = "native-cuda")]
type Backend = Cuda<f32, i32>;

struct Args {
    model_dir: PathBuf,
    prompt: String,
    output_path: PathBuf,
    width: usize,
    height: usize,
    steps: usize,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();

    let mut model_dir = PathBuf::from(".");
    let mut prompt = "A beautiful sunset over mountains".to_string();
    let mut output_path = PathBuf::from("output.png");
    let mut width = 512;
    let mut height = 512;
    let mut steps = 4;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model-dir" | "-m" => {
                if i + 1 < args.len() {
                    model_dir = PathBuf::from(&args[i + 1]);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--prompt" | "-p" => {
                if i + 1 < args.len() {
                    prompt = args[i + 1].clone();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--output" | "-o" => {
                if i + 1 < args.len() {
                    output_path = PathBuf::from(&args[i + 1]);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--width" | "-w" => {
                if i + 1 < args.len() {
                    width = args[i + 1].parse().unwrap_or(512);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--height" | "-h" => {
                if i + 1 < args.len() {
                    height = args[i + 1].parse().unwrap_or(512);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--steps" | "-s" => {
                if i + 1 < args.len() {
                    steps = args[i + 1].parse().unwrap_or(4);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--help" => {
                eprintln!("Usage: {} [OPTIONS]", args[0]);
                eprintln!("Options:");
                eprintln!("  --model-dir, -m <DIR>    Model directory");
                eprintln!("  --prompt, -p <TEXT>      Generation prompt");
                eprintln!("  --output, -o <FILE>      Output PNG file");
                eprintln!("  --width, -w <N>          Image width (default: 512)");
                eprintln!("  --height, -h <N>         Image height (default: 512)");
                eprintln!("  --steps, -s <N>          Inference steps (default: 4)");
                std::process::exit(0);
            }
            _ => {
                i += 1;
            }
        }
    }

    Args {
        model_dir,
        prompt,
        output_path,
        width,
        height,
        steps,
    }
}

fn main() {
    #[cfg(feature = "native-cuda")]
    {
        let args = parse_args();

        eprintln!("=== Z-Image Low-Memory Generation ===");
        eprintln!("Model dir: {:?}", args.model_dir);
        eprintln!("Prompt: {}", args.prompt);
        eprintln!("Output: {:?}", args.output_path);
        eprintln!("Size: {}x{}, Steps: {}", args.width, args.height, args.steps);
        eprintln!();

        let device = CudaDevice::new(0);
        eprintln!("Using CUDA device (native)");

        // === PHASE 1: Compute text embedding, then drop text encoder ===
        eprintln!("=== Phase 1: Computing text embedding ===");

        let tokenizer_path = args.model_dir.join("qwen3-tokenizer.json");
        eprintln!("Loading tokenizer from {:?}...", tokenizer_path);
        let tokenizer = match Qwen3Tokenizer::from_file(&tokenizer_path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Failed to load tokenizer: {}", e);
                return;
            }
        };
        eprintln!("Tokenizer loaded");

        // Compute embedding and save to CPU memory
        let embedding_data = {
            let te_path = args.model_dir.join("qwen3_4b_text_encoder_q8.bpk");
            eprintln!("Loading text encoder from {:?}...", te_path);
            let mut text_encoder: Qwen3Model<Backend> = Qwen3Config::z_image_text_encoder().init(&device);
            if let Err(e) = text_encoder.load_weights(&te_path) {
                eprintln!("Failed to load text encoder: {:?}", e);
                return;
            }
            eprintln!("Text encoder loaded (~4GB VRAM)");

            eprintln!("Computing embedding for: {}", args.prompt);
            let emb = match z_image::compute_prompt_embedding(&args.prompt, &tokenizer, &text_encoder, &device) {
                Ok(e) => e,
                Err(e) => {
                    eprintln!("Failed to compute embedding: {:?}", e);
                    return;
                }
            };
            let dims = emb.dims();
            eprintln!("Embedding computed, shape: {:?}", dims);

            // Move embedding data to CPU before dropping text encoder
            eprintln!("Moving embedding to CPU memory...");
            let data = emb.into_data();
            eprintln!("Embedding now in CPU memory");
            eprintln!("Text encoder will be dropped now to free VRAM...");
            (data, dims)
            // text_encoder and GPU embedding are dropped here
        };

        // Force VRAM cleanup after dropping text encoder
        eprintln!("Forcing VRAM cleanup...");
        <Backend as BackendTrait>::memory_cleanup(&device);
        let _ = <Backend as BackendTrait>::sync(&device);
        eprintln!("VRAM cleanup complete");

        eprintln!();
        eprintln!("=== Phase 2: Load transformer + autoencoder ===");

        // Load transformer
        let transformer_path = args.model_dir.join("z_image_turbo_q8.bpk");
        eprintln!("Loading transformer from {:?}...", transformer_path);
        let mut transformer: z_image::modules::transformer::ZImageModel<Backend> = ZImageModelConfig::default().init(&device);
        if let Err(e) = transformer.load_weights(&transformer_path) {
            eprintln!("Failed to load transformer: {:?}", e);
            return;
        }
        eprintln!("Transformer loaded (~6GB VRAM)");

        // Load autoencoder
        let ae_path = args.model_dir.join("ae.bpk");
        eprintln!("Loading autoencoder from {:?}...", ae_path);
        let mut ae: z_image::modules::ae::AutoEncoder<Backend> = AutoEncoderConfig::flux_ae().init(&device);
        if let Err(e) = ae.load_weights(&ae_path) {
            eprintln!("Failed to load autoencoder: {:?}", e);
            return;
        }
        eprintln!("Autoencoder loaded (~0.2GB VRAM)");

        eprintln!();
        eprintln!("=== Phase 3: Generate image ===");

        // Reconstruct embedding tensor on GPU
        let (data, dims) = embedding_data;
        eprintln!("Moving embedding back to GPU...");
        let embedding: Tensor<Backend, 3> = Tensor::<Backend, 1>::from_data(data, &device).reshape(dims);
        eprintln!("Embedding on GPU, shape: {:?}", embedding.dims());

        let opts = GenerateWithEmbeddingOpts {
            width: args.width,
            height: args.height,
            num_inference_steps: Some(args.steps),
            seed: None,
        };

        match z_image::generate_with_embedding(embedding, &args.output_path, &opts, &ae, &transformer, &device) {
            Ok(()) => {
                eprintln!("Success! Image saved to {:?}", args.output_path);
            }
            Err(e) => {
                eprintln!("Generation failed: {:?}", e);
            }
        }
    }

    #[cfg(not(feature = "native-cuda"))]
    {
        eprintln!("This binary requires the native-cuda feature");
        eprintln!("Build with: cargo build --release --no-default-features --features native-cuda --bin test_lowmem");
        std::process::exit(1);
    }
}
