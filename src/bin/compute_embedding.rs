//! Compute text embedding and save to disk
//!
//! This is the first step in the two-process low-memory generation flow:
//! 1. compute_embedding: Load text encoder, compute embedding, save to disk
//! 2. generate_from_embedding: Load transformer + AE, generate from saved embedding
//!
//! Embeddings are cached - same prompt will reuse existing embedding file.

use std::fs;
use std::io::Write;
use std::path::PathBuf;

#[cfg(feature = "native-cuda")]
use burn::backend::{Cuda, cuda::CudaDevice};
#[cfg(feature = "native-cuda")]
use burn::tensor::backend::Backend as BackendTrait;

use qwen3_burn::{Qwen3Config, Qwen3Model, Qwen3Tokenizer};

#[cfg(feature = "native-cuda")]
type Backend = Cuda<f32, i32>;

/// Embedding file format (simple binary):
/// - 4 bytes: magic "ZEMB"
/// - 4 bytes: version (u32 LE)
/// - 4 bytes: dim0 (u32 LE) - batch size
/// - 4 bytes: dim1 (u32 LE) - sequence length
/// - 4 bytes: dim2 (u32 LE) - hidden dim
/// - N bytes: f32 LE data
const MAGIC: &[u8; 4] = b"ZEMB";
const VERSION: u32 = 1;

struct Args {
    model_dir: PathBuf,
    prompt: String,
    output: PathBuf,
    force: bool,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();

    let mut model_dir = PathBuf::from(".");
    let mut prompt = String::new();
    let mut output = PathBuf::new();
    let mut force = false;

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
                    output = PathBuf::from(&args[i + 1]);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--force" | "-f" => {
                force = true;
                i += 1;
            }
            "--help" | "-h" => {
                eprintln!("Usage: {} [OPTIONS]", args[0]);
                eprintln!("Options:");
                eprintln!("  --model-dir, -m <DIR>    Model directory");
                eprintln!("  --prompt, -p <TEXT>      Text prompt to encode");
                eprintln!("  --output, -o <FILE>      Output embedding file (.zemb)");
                eprintln!("  --force, -f              Overwrite existing file");
                eprintln!("  --help, -h               Show this help");
                std::process::exit(0);
            }
            _ => {
                i += 1;
            }
        }
    }

    if prompt.is_empty() {
        eprintln!("Error: --prompt is required");
        std::process::exit(1);
    }

    if output.as_os_str().is_empty() {
        eprintln!("Error: --output is required");
        std::process::exit(1);
    }

    Args {
        model_dir,
        prompt,
        output,
        force,
    }
}

fn main() {
    #[cfg(feature = "native-cuda")]
    {
        let args = parse_args();

        // Check if output already exists
        if args.output.exists() && !args.force {
            eprintln!("Embedding file already exists: {:?}", args.output);
            eprintln!("Use --force to overwrite");
            std::process::exit(0);
        }

        eprintln!("=== Compute Text Embedding ===");
        eprintln!("Model dir: {:?}", args.model_dir);
        eprintln!("Prompt: {}", args.prompt);
        eprintln!("Output: {:?}", args.output);
        eprintln!();

        let device = CudaDevice::new(0);
        eprintln!("Using CUDA device");

        // Load tokenizer
        let tokenizer_path = args.model_dir.join("qwen3-tokenizer.json");
        eprintln!("Loading tokenizer from {:?}...", tokenizer_path);
        let tokenizer = match Qwen3Tokenizer::from_file(&tokenizer_path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Failed to load tokenizer: {}", e);
                std::process::exit(1);
            }
        };

        // Load text encoder
        let te_path = args.model_dir.join("qwen3_4b_text_encoder_q8.bpk");
        eprintln!("Loading text encoder from {:?}...", te_path);
        let mut text_encoder: Qwen3Model<Backend> = Qwen3Config::z_image_text_encoder().init(&device);
        if let Err(e) = text_encoder.load_weights(&te_path) {
            eprintln!("Failed to load text encoder: {:?}", e);
            std::process::exit(1);
        }
        eprintln!("Text encoder loaded");

        // Compute embedding
        eprintln!("Computing embedding...");
        let embedding = match z_image::compute_prompt_embedding(&args.prompt, &tokenizer, &text_encoder, &device) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("Failed to compute embedding: {:?}", e);
                std::process::exit(1);
            }
        };

        let dims = embedding.dims();
        eprintln!("Embedding shape: {:?}", dims);

        // Move to CPU and save
        eprintln!("Saving embedding to {:?}...", args.output);
        let data = embedding.into_data();
        let floats: Vec<f32> = data.as_slice().expect("Should be f32").to_vec();

        // Write embedding file
        let mut file = match fs::File::create(&args.output) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Failed to create file: {}", e);
                std::process::exit(1);
            }
        };

        // Write header
        file.write_all(MAGIC).unwrap();
        file.write_all(&VERSION.to_le_bytes()).unwrap();
        file.write_all(&(dims[0] as u32).to_le_bytes()).unwrap();
        file.write_all(&(dims[1] as u32).to_le_bytes()).unwrap();
        file.write_all(&(dims[2] as u32).to_le_bytes()).unwrap();

        // Write data
        for f in floats {
            file.write_all(&f.to_le_bytes()).unwrap();
        }

        // Also save the prompt alongside for reference
        let prompt_path = args.output.with_extension("txt");
        fs::write(&prompt_path, &args.prompt).ok();

        eprintln!("Done! Embedding saved to {:?}", args.output);
        eprintln!("Prompt saved to {:?}", prompt_path);

        // Cleanup
        drop(text_encoder);
        <Backend as BackendTrait>::memory_cleanup(&device);
    }

    #[cfg(not(feature = "native-cuda"))]
    {
        eprintln!("This binary requires the native-cuda feature");
        std::process::exit(1);
    }
}
