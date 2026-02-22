//! Debug attention mechanism

use std::path::PathBuf;

use burn::tensor::{DType, Tensor};

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
    let _model_dir = if args.len() > 1 {
        PathBuf::from(&args[1])
    } else {
        PathBuf::from(".")
    };

    eprintln!("Debug attention test");

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

    // Create test tensors for attention
    let batch_size = 1;
    let num_heads = 24;
    let seq_len = 64;
    let head_dim = 160;

    let query: Tensor<Backend, 4> = Tensor::zeros([batch_size, num_heads, seq_len, head_dim], &device);
    let key: Tensor<Backend, 4> = Tensor::zeros([batch_size, num_heads, seq_len, head_dim], &device);
    let value: Tensor<Backend, 4> = Tensor::zeros([batch_size, num_heads, seq_len, head_dim], &device);

    eprintln!("Query shape: {:?}", query.dims());
    eprintln!("Key shape: {:?}", key.dims());
    eprintln!("Value shape: {:?}", value.dims());

    // Test basic matmul
    let scores = query.clone().matmul(key.clone().transpose());
    eprintln!("Scores shape: {:?}", scores.dims());

    eprintln!("Debug complete");
}
