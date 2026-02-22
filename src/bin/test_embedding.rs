//! Test text embedding with Qwen3 model

use std::path::PathBuf;

use burn::module::Module;
use burn::tensor::{Int, Tensor};
use qwen3_burn::{Qwen3Config, Qwen3Model, Qwen3Tokenizer};

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
    let model_dir = if args.len() > 1 {
        PathBuf::from(&args[1])
    } else {
        PathBuf::from(".")
    };

    eprintln!("Model dir: {:?}", model_dir);

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
    eprintln!("Using GPU device");

    // Load tokenizer
    let tokenizer_path = model_dir.join("qwen3-tokenizer.json");
    eprintln!("Loading tokenizer from {:?}...", tokenizer_path);
    let tokenizer = match Qwen3Tokenizer::from_file(&tokenizer_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Failed to load tokenizer: {}", e);
            return;
        }
    };

    // Load model
    let te_bpk = model_dir.join("qwen3_4b_text_encoder.bpk");
    let te_safetensors = model_dir.join("qwen3_4b_text_encoder.safetensors");
    let model_path = if te_bpk.exists() { te_bpk } else { te_safetensors };
    eprintln!("Loading text encoder from {:?}...", model_path);

    let mut model: Qwen3Model<Backend> = Qwen3Config::z_image_text_encoder().init(&device);
    if let Err(e) = model.load_weights(&model_path) {
        eprintln!("Failed to load model: {:?}", e);
        return;
    }
    eprintln!("Model loaded");

    // Test embedding
    let test_prompt = "A beautiful sunset over mountains";
    eprintln!("\nTest prompt: {}", test_prompt);

    let formatted = tokenizer.apply_chat_template(test_prompt);
    let (input_ids, attention_mask) = match tokenizer.encode(&formatted) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Tokenization failed: {}", e);
            return;
        }
    };

    eprintln!("Input tokens: {}", input_ids.len());

    let seq_len = input_ids.len();
    let input_tensor: Tensor<Backend, 2, Int> =
        Tensor::<Backend, 1, Int>::from_data(
            input_ids.iter().map(|&x| x as i64).collect::<Vec<_>>().as_slice(),
            &device,
        ).reshape([1, seq_len]);

    let mask_tensor: Tensor<Backend, 2> = Tensor::<Backend, 1>::from_data(
        attention_mask.iter().map(|&b| if b { 1.0f32 } else { 0.0f32 }).collect::<Vec<_>>().as_slice(),
        &device,
    ).reshape([1, seq_len]);

    eprintln!("Computing embedding...");
    let attention_mask_bool = mask_tensor.greater_elem(0.5);
    let embeddings = model.forward(input_tensor, Some(attention_mask_bool));
    eprintln!("Got {} embedding tensors", embeddings.len());
    if let Some(first) = embeddings.first() {
        eprintln!("First embedding shape: {:?}", first.dims());
    }
}
