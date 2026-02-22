#![recursion_limit = "256"]
//! Convert safetensors models to Burn's .bpk format
//!
//! Usage:
//!   cargo run --release --bin convert_models -- --ae /path/to/ae.safetensors --output /path/to/output/
//!   cargo run --release --bin convert_models -- --text-encoder /path/to/qwen3_4b.safetensors --output /path/to/output/
//!   cargo run --release --bin convert_models -- --all /path/to/model_dir --output /path/to/output/

use std::path::PathBuf;

use burn::module::Module;
use burn::store::{BurnpackWriter, Collector};

#[cfg(any(feature = "metal", feature = "cuda"))]
use burn::backend::candle::{Candle, CandleDevice};
#[cfg(any(feature = "metal", feature = "cuda"))]
type Backend = Candle<half::bf16, i64>;

#[cfg(any(feature = "wgpu", feature = "wgpu-metal", feature = "vulkan"))]
#[cfg(not(any(feature = "metal", feature = "cuda")))]
use burn::backend::wgpu::{Wgpu, WgpuDevice};
#[cfg(any(feature = "wgpu", feature = "wgpu-metal", feature = "vulkan"))]
#[cfg(not(any(feature = "metal", feature = "cuda")))]
type Backend = Wgpu<f32, i32>;

#[cfg(not(any(feature = "metal", feature = "cuda", feature = "wgpu", feature = "wgpu-metal", feature = "vulkan")))]
use burn::backend::ndarray::{NdArray, NdArrayDevice};
#[cfg(not(any(feature = "metal", feature = "cuda", feature = "wgpu", feature = "wgpu-metal", feature = "vulkan")))]
type Backend = NdArray<f32>;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 4 {
        eprintln!("Convert Z-Image models to Burn's .bpk format\n");
        eprintln!("Usage:");
        eprintln!("  {} --ae <ae.safetensors> --output <output_dir>", args[0]);
        eprintln!("  {} --text-encoder <qwen3_4b.safetensors> --output <output_dir>", args[0]);
        eprintln!("  {} --all <model_dir> --output <output_dir>", args[0]);
        std::process::exit(1);
    }

    #[cfg(feature = "metal")]
    let device = CandleDevice::metal(0);
    #[cfg(feature = "cuda")]
    #[cfg(not(feature = "metal"))]
    let device = CandleDevice::cuda(0);
    #[cfg(any(feature = "wgpu", feature = "wgpu-metal", feature = "vulkan"))]
    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    let device = WgpuDevice::default();
    #[cfg(not(any(feature = "metal", feature = "cuda", feature = "wgpu", feature = "wgpu-metal", feature = "vulkan")))]
    let device = NdArrayDevice::Cpu;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--ae" => {
                let input = PathBuf::from(&args[i + 1]);
                let output_dir = find_output_dir(&args);
                convert_autoencoder(&input, &output_dir, &device);
                i += 2;
            }
            "--text-encoder" => {
                let input = PathBuf::from(&args[i + 1]);
                let output_dir = find_output_dir(&args);
                convert_text_encoder(&input, &output_dir, &device);
                i += 2;
            }
            "--all" => {
                let model_dir = PathBuf::from(&args[i + 1]);
                let output_dir = find_output_dir(&args);
                convert_all(&model_dir, &output_dir, &device);
                i += 2;
            }
            "--output" => {
                i += 2;
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                i += 1;
            }
        }
    }
}

fn find_output_dir(args: &[String]) -> PathBuf {
    for i in 0..args.len() - 1 {
        if args[i] == "--output" {
            return PathBuf::from(&args[i + 1]);
        }
    }
    PathBuf::from(".")
}

fn convert_autoencoder(input: &PathBuf, output_dir: &PathBuf, device: &<Backend as burn::prelude::Backend>::Device) {
    use z_image::modules::ae::AutoEncoderConfig;

    eprintln!("=== Converting Autoencoder ===");
    eprintln!("Input: {:?}", input);

    let mut ae = AutoEncoderConfig::flux_ae().init::<Backend>(device);

    eprintln!("Loading weights from safetensors...");
    if let Err(e) = ae.load_weights(input) {
        eprintln!("Failed to load weights: {:?}", e);
        return;
    }
    eprintln!("Weights loaded successfully");

    eprintln!("Collecting tensors...");
    let mut collector = Collector::default();
    ae.visit(&mut collector);
    let snapshots = collector.into_tensors();
    eprintln!("Collected {} tensors", snapshots.len());

    std::fs::create_dir_all(output_dir).ok();

    let output_path = output_dir.join("ae.bpk");
    eprintln!("Writing to {:?}...", output_path);

    let writer = BurnpackWriter::new(snapshots)
        .with_metadata("model_type", "flux_autoencoder")
        .with_metadata("converted_from", "ae.safetensors");

    match writer.write_to_file(&output_path) {
        Ok(_) => {
            let size = std::fs::metadata(&output_path).map(|m| m.len()).unwrap_or(0);
            eprintln!("Successfully saved ae.bpk ({:.1} MB)", size as f64 / 1024.0 / 1024.0);
        }
        Err(e) => eprintln!("Failed to save: {:?}", e),
    }
}

fn convert_text_encoder(input: &PathBuf, output_dir: &PathBuf, device: &<Backend as burn::prelude::Backend>::Device) {
    use qwen3_burn::{Qwen3Config, Qwen3Model};

    eprintln!("=== Converting Qwen3-4B Text Encoder (Z-Image variant) ===");
    eprintln!("Input: {:?}", input);

    let config = Qwen3Config::z_image_text_encoder();
    let mut model: Qwen3Model<Backend> = config.init(device);

    eprintln!("Loading weights from safetensors...");
    if let Err(e) = model.load_weights(input) {
        eprintln!("Failed to load weights: {:?}", e);
        return;
    }
    eprintln!("Weights loaded successfully");

    eprintln!("Collecting tensors...");
    let mut collector = Collector::default();
    model.visit(&mut collector);
    let snapshots = collector.into_tensors();
    eprintln!("Collected {} tensors", snapshots.len());

    std::fs::create_dir_all(output_dir).ok();

    let output_path = output_dir.join("qwen3_4b_text_encoder.bpk");
    eprintln!("Writing to {:?}...", output_path);

    let writer = BurnpackWriter::new(snapshots)
        .with_metadata("model_type", "qwen3_4b_text_encoder")
        .with_metadata("converted_from", "qwen3_4b_text_encoder.safetensors");

    match writer.write_to_file(&output_path) {
        Ok(_) => {
            let size = std::fs::metadata(&output_path).map(|m| m.len()).unwrap_or(0);
            eprintln!("Successfully saved qwen3_4b_text_encoder.bpk ({:.1} MB)", size as f64 / 1024.0 / 1024.0);
        }
        Err(e) => eprintln!("Failed to save: {:?}", e),
    }
}

fn convert_all(model_dir: &PathBuf, output_dir: &PathBuf, device: &<Backend as burn::prelude::Backend>::Device) {
    eprintln!("=== Converting All Models ===");
    eprintln!("Source directory: {:?}", model_dir);
    eprintln!("Output directory: {:?}", output_dir);

    std::fs::create_dir_all(output_dir).ok();

    let ae_path = model_dir.join("ae.safetensors");
    if ae_path.exists() {
        convert_autoencoder(&ae_path, output_dir, device);
    } else {
        eprintln!("ae.safetensors not found at {:?}", ae_path);
    }

    eprintln!();

    let te_path = model_dir.join("qwen3_4b_text_encoder.safetensors");
    if te_path.exists() {
        convert_text_encoder(&te_path, output_dir, device);
    } else {
        eprintln!("qwen3_4b_text_encoder.safetensors not found at {:?}", te_path);
    }

    eprintln!();

    let tokenizer_path = model_dir.join("qwen3-tokenizer.json");
    if tokenizer_path.exists() {
        let dest = output_dir.join("qwen3-tokenizer.json");
        if let Err(e) = std::fs::copy(&tokenizer_path, &dest) {
            eprintln!("Failed to copy tokenizer: {:?}", e);
        } else {
            eprintln!("Copied tokenizer to {:?}", dest);
        }
    }

    eprintln!("\n=== Conversion Complete ===");
    eprintln!("Output files in {:?}:", output_dir);
    if let Ok(entries) = std::fs::read_dir(output_dir) {
        for entry in entries.flatten() {
            if let Ok(metadata) = entry.metadata() {
                let size_mb = metadata.len() as f64 / (1024.0 * 1024.0);
                eprintln!("  {:?} ({:.1} MB)", entry.file_name(), size_mb);
            }
        }
    }
}
