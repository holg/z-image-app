//! Test tokenizer loading and encoding

use std::path::PathBuf;
use qwen3_burn::Qwen3Tokenizer;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let tokenizer_path = if args.len() > 1 {
        PathBuf::from(&args[1])
    } else {
        PathBuf::from("tokenizer.json")
    };

    eprintln!("Loading tokenizer from {:?}...", tokenizer_path);

    let tokenizer = match Qwen3Tokenizer::from_file(&tokenizer_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Failed to load tokenizer: {}", e);
            return;
        }
    };

    eprintln!("Tokenizer loaded successfully!");

    let test_text = "Hello, how are you today?";
    eprintln!("\nTest text: {}", test_text);

    let formatted = tokenizer.apply_chat_template(test_text);
    eprintln!("Formatted: {}", formatted);

    match tokenizer.encode(&formatted) {
        Ok((ids, mask)) => {
            eprintln!("Token IDs ({} tokens): {:?}", ids.len(), &ids[..ids.len().min(20)]);
            eprintln!("Attention mask: {:?}", &mask[..mask.len().min(20)]);
        }
        Err(e) => {
            eprintln!("Encoding failed: {}", e);
        }
    }
}
