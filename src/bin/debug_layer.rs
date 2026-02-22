//! Debug transformer layers

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
    let model_dir = if args.len() > 1 {
        PathBuf::from(&args[1])
    } else {
        PathBuf::from(".")
    };

    eprintln!("Debug layer test");
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

    // Create test tensor
    let test_tensor: Tensor<Backend, 3> = Tensor::zeros([1, 64, 3840], &device);
    eprintln!("Test tensor shape: {:?}", test_tensor.dims());
    eprintln!("Test tensor dtype: {:?}", test_tensor.dtype());

    // Test dtype conversion
    let f32_tensor = test_tensor.clone().cast(DType::F32);
    eprintln!("F32 tensor dtype: {:?}", f32_tensor.dtype());

    eprintln!("Debug complete");
}
