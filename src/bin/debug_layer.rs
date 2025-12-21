//! Debug transformer layers

use std::path::PathBuf;

use burn::backend::candle::{Candle, CandleDevice};
use burn::tensor::{DType, Tensor};
use half::bf16;

type Backend = Candle<bf16, i64>;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_dir = if args.len() > 1 {
        PathBuf::from(&args[1])
    } else {
        PathBuf::from(".")
    };

    eprintln!("Debug layer test");
    eprintln!("Model dir: {:?}", model_dir);

    let device = CandleDevice::metal(0);

    // Create test tensor
    let test_tensor: Tensor<Backend, 3> = Tensor::zeros([1, 64, 3840], &device);
    eprintln!("Test tensor shape: {:?}", test_tensor.dims());
    eprintln!("Test tensor dtype: {:?}", test_tensor.dtype());

    // Test dtype conversion
    let f32_tensor = test_tensor.clone().cast(DType::F32);
    eprintln!("F32 tensor dtype: {:?}", f32_tensor.dtype());

    eprintln!("Debug complete");
}
