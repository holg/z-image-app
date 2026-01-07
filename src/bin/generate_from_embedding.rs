//! Generate image from pre-computed embedding
//!
//! This is the second step in the two-process low-memory generation flow:
//! 1. compute_embedding: Load text encoder, compute embedding, save to disk
//! 2. generate_from_embedding: Load transformer + AE, generate from saved embedding
//!
//! This process only loads transformer (~6GB) + autoencoder (~0.2GB), fitting in 8GB VRAM.

use std::fs;
use std::io::Read;
use std::path::PathBuf;

#[cfg(feature = "native-cuda")]
use burn::backend::{Cuda, cuda::CudaDevice};
#[cfg(feature = "native-cuda")]
use burn::Tensor;

use z_image::{GenerateWithEmbeddingOpts, modules::ae::AutoEncoderConfig, modules::transformer::ZImageModelConfig};

#[cfg(feature = "native-cuda")]
type Backend = Cuda<f32, i32>;

const MAGIC: &[u8; 4] = b"ZEMB";

struct Args {
    model_dir: PathBuf,
    embedding: PathBuf,
    output: PathBuf,
    width: usize,
    height: usize,
    steps: usize,
    seed: Option<u64>,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();

    let mut model_dir = PathBuf::from(".");
    let mut embedding = PathBuf::new();
    let mut output = PathBuf::from("output.png");
    let mut width = 512;
    let mut height = 512;
    let mut steps = 4;
    let mut seed = None;

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
            "--embedding" | "-e" => {
                if i + 1 < args.len() {
                    embedding = PathBuf::from(&args[i + 1]);
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
            "--seed" => {
                if i + 1 < args.len() {
                    seed = args[i + 1].parse().ok();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--help" => {
                eprintln!("Usage: {} [OPTIONS]", args[0]);
                eprintln!("Options:");
                eprintln!("  --model-dir, -m <DIR>    Model directory");
                eprintln!("  --embedding, -e <FILE>   Input embedding file (.zemb)");
                eprintln!("  --output, -o <FILE>      Output PNG file");
                eprintln!("  --width, -w <N>          Image width (default: 512)");
                eprintln!("  --height, -h <N>         Image height (default: 512)");
                eprintln!("  --steps, -s <N>          Inference steps (default: 4)");
                eprintln!("  --seed <N>               Random seed (optional)");
                eprintln!("  --help                   Show this help");
                std::process::exit(0);
            }
            _ => {
                i += 1;
            }
        }
    }

    if embedding.as_os_str().is_empty() {
        eprintln!("Error: --embedding is required");
        std::process::exit(1);
    }

    Args {
        model_dir,
        embedding,
        output,
        width,
        height,
        steps,
        seed,
    }
}

#[cfg(feature = "native-cuda")]
fn load_embedding(path: &PathBuf, device: &CudaDevice) -> Result<Tensor<Backend, 3>, String> {
    let mut file = fs::File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;

    // Read and verify magic
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic).map_err(|e| format!("Failed to read magic: {}", e))?;
    if &magic != MAGIC {
        return Err("Invalid embedding file format".to_string());
    }

    // Read version
    let mut version_bytes = [0u8; 4];
    file.read_exact(&mut version_bytes).map_err(|e| format!("Failed to read version: {}", e))?;
    let version = u32::from_le_bytes(version_bytes);
    if version != 1 {
        return Err(format!("Unsupported version: {}", version));
    }

    // Read dimensions
    let mut dim_bytes = [0u8; 4];
    file.read_exact(&mut dim_bytes).map_err(|e| format!("Failed to read dim0: {}", e))?;
    let dim0 = u32::from_le_bytes(dim_bytes) as usize;

    file.read_exact(&mut dim_bytes).map_err(|e| format!("Failed to read dim1: {}", e))?;
    let dim1 = u32::from_le_bytes(dim_bytes) as usize;

    file.read_exact(&mut dim_bytes).map_err(|e| format!("Failed to read dim2: {}", e))?;
    let dim2 = u32::from_le_bytes(dim_bytes) as usize;

    // Read data
    let num_floats = dim0 * dim1 * dim2;
    let mut floats = Vec::with_capacity(num_floats);
    let mut float_bytes = [0u8; 4];

    for _ in 0..num_floats {
        file.read_exact(&mut float_bytes).map_err(|e| format!("Failed to read data: {}", e))?;
        floats.push(f32::from_le_bytes(float_bytes));
    }

    // Create tensor
    let tensor: Tensor<Backend, 1> = Tensor::from_floats(floats.as_slice(), device);
    let tensor: Tensor<Backend, 3> = tensor.reshape([dim0, dim1, dim2]);

    Ok(tensor)
}

fn main() {
    #[cfg(feature = "native-cuda")]
    {
        let args = parse_args();

        eprintln!("=== Generate From Embedding ===");
        eprintln!("Model dir: {:?}", args.model_dir);
        eprintln!("Embedding: {:?}", args.embedding);
        eprintln!("Output: {:?}", args.output);
        eprintln!("Size: {}x{}, Steps: {}", args.width, args.height, args.steps);
        if let Some(seed) = args.seed {
            eprintln!("Seed: {}", seed);
        }
        eprintln!();

        let device = CudaDevice::new(0);
        eprintln!("Using CUDA device");

        // Enable memory optimizations for 8GB VRAM
        z_image::set_attention_slice_size(4);
        z_image::set_attention_seq_chunk_size(512);
        eprintln!("Memory optimizations enabled (head_slice=4, seq_chunk=512)");

        // Load embedding from file
        eprintln!("Loading embedding from {:?}...", args.embedding);
        let embedding = match load_embedding(&args.embedding, &device) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("Failed to load embedding: {}", e);
                std::process::exit(1);
            }
        };
        eprintln!("Embedding loaded, shape: {:?}", embedding.dims());

        // Show prompt if available
        let prompt_path = args.embedding.with_extension("txt");
        if let Ok(prompt) = fs::read_to_string(&prompt_path) {
            eprintln!("Original prompt: {}", prompt.trim());
        }
        eprintln!();

        // Load transformer
        let transformer_path = args.model_dir.join("z_image_turbo_q8.bpk");
        eprintln!("Loading transformer from {:?}...", transformer_path);
        let mut transformer: z_image::modules::transformer::ZImageModel<Backend> = ZImageModelConfig::default().init(&device);
        if let Err(e) = transformer.load_weights(&transformer_path) {
            eprintln!("Failed to load transformer: {:?}", e);
            std::process::exit(1);
        }
        eprintln!("Transformer loaded (~6GB VRAM)");

        // Load autoencoder
        let ae_path = args.model_dir.join("ae.bpk");
        eprintln!("Loading autoencoder from {:?}...", ae_path);
        let mut ae: z_image::modules::ae::AutoEncoder<Backend> = AutoEncoderConfig::flux_ae().init(&device);
        if let Err(e) = ae.load_weights(&ae_path) {
            eprintln!("Failed to load autoencoder: {:?}", e);
            std::process::exit(1);
        }
        eprintln!("Autoencoder loaded (~0.2GB VRAM)");
        eprintln!();

        // Generate
        eprintln!("Generating image...");
        let opts = GenerateWithEmbeddingOpts {
            width: args.width,
            height: args.height,
            num_inference_steps: Some(args.steps),
            seed: args.seed,
        };

        match z_image::generate_with_embedding(embedding, &args.output, &opts, &ae, &transformer, &device) {
            Ok(()) => {
                eprintln!("Success! Image saved to {:?}", args.output);
            }
            Err(e) => {
                eprintln!("Generation failed: {:?}", e);
                std::process::exit(1);
            }
        }
    }

    #[cfg(not(feature = "native-cuda"))]
    {
        eprintln!("This binary requires the native-cuda feature");
        std::process::exit(1);
    }
}
