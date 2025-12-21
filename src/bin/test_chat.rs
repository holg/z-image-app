//! Test text generation with Qwen3-0.6B
//!
//! Usage:
//!   cargo run --release --bin test_chat -- --model-dir /path/to/qwen3-0.6b

use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use burn::backend::candle::{Candle, CandleDevice};
use burn::module::Module;
use burn::tensor::{Int, Tensor};
use half::bf16;
use qwen3_burn::{Qwen3Config, Qwen3ForCausalLM, Qwen3Tokenizer};

type Backend = Candle<bf16, i64>;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut model_dir = PathBuf::from(".");

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
            "--help" => {
                eprintln!("Usage: {} --model-dir <DIR>", args[0]);
                return;
            }
            _ => {
                i += 1;
            }
        }
    }

    eprintln!("=== Qwen3-0.6B Chat Test ===");
    eprintln!("Model dir: {:?}", model_dir);

    let device = CandleDevice::metal(0);
    eprintln!("Using Metal device");

    // Load tokenizer
    let tokenizer_path = model_dir.join("tokenizer.json");
    eprintln!("Loading tokenizer from {:?}...", tokenizer_path);
    let tokenizer = match Qwen3Tokenizer::from_file(&tokenizer_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Failed to load tokenizer: {}", e);
            return;
        }
    };
    eprintln!("Tokenizer loaded");

    // Load model
    let bpk_path = model_dir.join("model.bpk");
    let safetensors_path = model_dir.join("model.safetensors");
    let model_path = if bpk_path.exists() { bpk_path } else { safetensors_path };
    eprintln!("Loading model from {:?}...", model_path);

    let mut model: Qwen3ForCausalLM<Backend> = Qwen3Config::qwen3_0_6b().init_causal_lm(&device);
    if let Err(e) = model.load_weights(&model_path) {
        eprintln!("Failed to load model: {:?}", e);
        return;
    }
    eprintln!("Model loaded");
    eprintln!();
    eprintln!("Type your message (or 'quit' to exit):");

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("> ");
        stdout.flush().unwrap();

        let mut input = String::new();
        stdin.lock().read_line(&mut input).unwrap();
        let input = input.trim();

        if input.is_empty() {
            continue;
        }
        if input == "quit" || input == "exit" {
            break;
        }

        // Format with chat template
        let formatted = tokenizer.apply_chat_template(input);

        // Tokenize
        let (input_ids, _) = match tokenizer.encode_no_pad(&formatted) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Tokenization failed: {}", e);
                continue;
            }
        };

        eprintln!("[{} input tokens]", input_ids.len());

        // Create tensor
        let input_ids_i64: Vec<i64> = input_ids.iter().map(|&x| x as i64).collect();
        let input_tensor: Tensor<Backend, 2, Int> =
            Tensor::<Backend, 1, Int>::from_data(input_ids_i64.as_slice(), &device)
                .reshape([1, input_ids.len()]);

        // Generate
        let output = model.generate_with_cache(input_tensor, 256, 0.7, 0.9, 50);

        // Decode
        let output_data = output.into_data();
        let output_ids: Vec<u32> = output_data
            .as_slice::<i64>()
            .unwrap_or(&[])
            .iter()
            .skip(input_ids.len())
            .map(|&x| x as u32)
            .collect();

        let text = tokenizer.decode(&output_ids).unwrap_or_default();

        // Clean up
        let mut clean_text = text;
        if let Some(pos) = clean_text.find("<|im_end|>") {
            clean_text = clean_text[..pos].to_string();
        }
        if let Some(pos) = clean_text.find("<|endoftext|>") {
            clean_text = clean_text[..pos].to_string();
        }

        println!("\nQwen3: {}\n", clean_text.trim());
    }

    eprintln!("Goodbye!");
}
