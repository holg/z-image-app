//! Debug attention mechanism

use std::path::PathBuf;

use burn::backend::candle::{Candle, CandleDevice};
use burn::tensor::{DType, Tensor};
use half::bf16;

type Backend = Candle<bf16, i64>;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let _model_dir = if args.len() > 1 {
        PathBuf::from(&args[1])
    } else {
        PathBuf::from(".")
    };

    eprintln!("Debug attention test");

    let device = CandleDevice::metal(0);

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
