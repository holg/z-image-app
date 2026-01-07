//! Test image generation
//!
//! Usage:
//!   cargo run --release --bin test_generate -- --model-dir /path/to/models --prompt "A cat" --output output.png

use std::path::PathBuf;

// Backend imports based on feature flags
#[cfg(any(feature = "cuda", feature = "metal"))]
use burn::backend::candle::{Candle, CandleDevice};
#[cfg(feature = "native-cuda")]
use burn::backend::{Cuda, cuda::CudaDevice};
#[cfg(any(feature = "vulkan", feature = "wgpu", feature = "wgpu-metal"))]
use burn::backend::{Wgpu, wgpu::WgpuDevice};

use burn::module::Module;
#[cfg(any(feature = "cuda", feature = "metal"))]
use half::bf16;
use qwen3_burn::{Qwen3Config, Qwen3Model, Qwen3Tokenizer};
use z_image::{GenerateFromTextOpts, modules::ae::AutoEncoderConfig, modules::transformer::ZImageModelConfig};

// Backend type based on feature flags
#[cfg(any(feature = "cuda", feature = "metal"))]
type Backend = Candle<bf16, i64>;
#[cfg(feature = "native-cuda")]
type Backend = Cuda<f32, i32>;
#[cfg(any(feature = "vulkan", feature = "wgpu", feature = "wgpu-metal"))]
type Backend = Wgpu;

struct Args {
    model_dir: PathBuf,
    prompt: String,
    output_path: PathBuf,
    width: usize,
    height: usize,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();

    let mut model_dir = PathBuf::from(".");
    let mut prompt = "A beautiful sunset over mountains".to_string();
    let mut output_path = PathBuf::from("output.png");
    let mut width = 512;
    let mut height = 512;

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
            "--help" => {
                eprintln!("Usage: {} [OPTIONS]", args[0]);
                eprintln!("Options:");
                eprintln!("  --model-dir, -m <DIR>    Model directory");
                eprintln!("  --prompt, -p <TEXT>      Generation prompt");
                eprintln!("  --output, -o <FILE>      Output PNG file");
                eprintln!("  --width, -w <N>          Image width (default: 512)");
                eprintln!("  --height, -h <N>         Image height (default: 512)");
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
    }
}

fn main() {
    let args = parse_args();

    eprintln!("=== Z-Image Test Generate ===");
    eprintln!("Model dir: {:?}", args.model_dir);
    eprintln!("Prompt: {}", args.prompt);
    eprintln!("Output: {:?}", args.output_path);
    eprintln!("Size: {}x{}", args.width, args.height);

    // Device selection based on backend
    #[cfg(feature = "metal")]
    let device = {
        eprintln!("Using Metal device");
        CandleDevice::metal(0)
    };
    #[cfg(feature = "cuda")]
    let device = {
        eprintln!("Using CUDA device (Candle)");
        CandleDevice::cuda(0)
    };
    #[cfg(feature = "native-cuda")]
    let device = {
        eprintln!("Using CUDA device (native)");
        CudaDevice::new(0)
    };
    #[cfg(any(feature = "vulkan", feature = "wgpu", feature = "wgpu-metal"))]
    let device = {
        eprintln!("Using WGPU device");
        WgpuDevice::default()
    };

    // Load tokenizer
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

    // Load text encoder - use Q8 version if available
    let te_q8 = args.model_dir.join("qwen3_4b_text_encoder_q8.bpk");
    let te_bpk = args.model_dir.join("qwen3_4b_text_encoder.bpk");
    let te_safetensors = args.model_dir.join("qwen3_4b_text_encoder.safetensors");
    let te_path = if te_q8.exists() { te_q8 } else if te_bpk.exists() { te_bpk } else { te_safetensors };
    eprintln!("Loading text encoder from {:?}...", te_path);
    let mut text_encoder: Qwen3Model<Backend> = Qwen3Config::z_image_text_encoder().init(&device);
    if let Err(e) = text_encoder.load_weights(&te_path) {
        eprintln!("Failed to load text encoder: {:?}", e);
        return;
    }
    eprintln!("Text encoder loaded");

    // Load transformer - use Q8 version if available
    let transformer_q8 = args.model_dir.join("z_image_turbo_q8.bpk");
    let transformer_bf16 = args.model_dir.join("z_image_turbo_bf16.bpk");
    let transformer_path = if transformer_q8.exists() { transformer_q8 } else { transformer_bf16 };
    eprintln!("Loading transformer from {:?}...", transformer_path);
    let mut transformer: z_image::modules::transformer::ZImageModel<Backend> = ZImageModelConfig::default().init(&device);
    if let Err(e) = transformer.load_weights(&transformer_path) {
        eprintln!("Failed to load transformer: {:?}", e);
        return;
    }
    eprintln!("Transformer loaded");

    // Load autoencoder
    let ae_bpk = args.model_dir.join("ae.bpk");
    let ae_safetensors = args.model_dir.join("ae.safetensors");
    let ae_path = if ae_bpk.exists() { ae_bpk } else { ae_safetensors };
    eprintln!("Loading autoencoder from {:?}...", ae_path);
    let mut ae: z_image::modules::ae::AutoEncoder<Backend> = AutoEncoderConfig::flux_ae().init(&device);
    if let Err(e) = ae.load_weights(&ae_path) {
        eprintln!("Failed to load autoencoder: {:?}", e);
        return;
    }
    eprintln!("Autoencoder loaded");

    // Generate
    eprintln!("Generating image for: {}", args.prompt);
    let opts = GenerateFromTextOpts {
        prompt: args.prompt.clone(),
        out_path: args.output_path.clone(),
        width: args.width,
        height: args.height,
        num_inference_steps: None,
        seed: None,
    };

    match z_image::generate_from_text(&opts, &tokenizer, &text_encoder, &ae, &transformer, &device) {
        Ok(()) => {
            eprintln!("Success! Image saved to {:?}", args.output_path);
        }
        Err(e) => {
            eprintln!("Generation failed: {:?}", e);
        }
    }
}
