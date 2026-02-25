#![recursion_limit = "256"]
//! Z-Image Server - HTTP API for image generation
//!
//! Build:
//!   cargo build --release --features "server,metal"
//!
//! Run:
//!   z-image-server --model-dir /path/to/models --port 8080

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize};
use std::sync::{Arc, RwLock};

use axum::routing::{get, post};
use axum::Router;
use tower_http::cors::CorsLayer;

#[path = "../server/mod.rs"]
mod server;

use server::handlers;
use server::queue;
use server::state::AppState;

// Backend selection based on compile-time features (same pattern as gui.rs / lib.rs)
#[cfg(feature = "metal")]
mod backend {
    use burn::backend::candle::{Candle, CandleDevice};
    use half::bf16;
    pub type Backend = Candle<bf16, i64>;
    pub type Device = CandleDevice;
    pub fn create_device() -> Device {
        CandleDevice::metal(0)
    }
    pub const BACKEND_NAME: &str = "Metal (Candle)";
}

#[cfg(feature = "cuda")]
#[cfg(not(feature = "metal"))]
mod backend {
    use burn::backend::candle::{Candle, CandleDevice};
    use half::bf16;
    pub type Backend = Candle<bf16, i64>;
    pub type Device = CandleDevice;
    pub fn create_device() -> Device {
        CandleDevice::cuda(0)
    }
    pub const BACKEND_NAME: &str = "CUDA (Candle)";
}

#[cfg(any(feature = "wgpu", feature = "wgpu-metal", feature = "vulkan"))]
#[cfg(not(any(feature = "metal", feature = "cuda")))]
mod backend {
    use burn::backend::wgpu::{Wgpu, WgpuDevice};
    pub type Backend = Wgpu<f32, i32>;
    pub type Device = WgpuDevice;
    pub fn create_device() -> Device {
        WgpuDevice::default()
    }
    pub const BACKEND_NAME: &str = "WGPU";
}

#[cfg(feature = "cpu")]
#[cfg(not(any(
    feature = "metal",
    feature = "cuda",
    feature = "wgpu",
    feature = "wgpu-metal",
    feature = "vulkan"
)))]
mod backend {
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    pub type Backend = NdArray<f32>;
    pub type Device = NdArrayDevice;
    pub fn create_device() -> Device {
        NdArrayDevice::Cpu
    }
    pub const BACKEND_NAME: &str = "CPU (NdArray)";
}

#[cfg(not(any(
    feature = "metal",
    feature = "cuda",
    feature = "wgpu",
    feature = "wgpu-metal",
    feature = "vulkan",
    feature = "cpu"
)))]
mod backend {
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    pub type Backend = NdArray<f32>;
    pub type Device = NdArrayDevice;
    pub fn create_device() -> Device {
        NdArrayDevice::Cpu
    }
    pub const BACKEND_NAME: &str = "CPU (fallback)";
}

type Backend = backend::Backend;

fn parse_args() -> (PathBuf, u16, PathBuf) {
    let args: Vec<String> = std::env::args().collect();
    let mut model_dir = PathBuf::from("models");
    let mut port: u16 = 8080;
    let mut output_dir = PathBuf::from("output");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model-dir" | "-m" => {
                i += 1;
                if i < args.len() {
                    model_dir = PathBuf::from(&args[i]);
                }
            }
            "--port" | "-p" => {
                i += 1;
                if i < args.len() {
                    port = args[i].parse().unwrap_or(8080);
                }
            }
            "--output-dir" | "-o" => {
                i += 1;
                if i < args.len() {
                    output_dir = PathBuf::from(&args[i]);
                }
            }
            "--help" | "-h" => {
                eprintln!("z-image-server - HTTP API for Z-Image generation");
                eprintln!();
                eprintln!("Usage: z-image-server [OPTIONS]");
                eprintln!();
                eprintln!("Options:");
                eprintln!("  -m, --model-dir <DIR>   Model directory (default: models)");
                eprintln!("  -p, --port <PORT>       Port to listen on (default: 8080)");
                eprintln!("  -o, --output-dir <DIR>  Output directory for images (default: output)");
                eprintln!("  -h, --help              Show this help");
                std::process::exit(0);
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
            }
        }
        i += 1;
    }

    (model_dir, port, output_dir)
}

#[tokio::main]
async fn main() {
    let (model_dir, port, output_dir) = parse_args();

    eprintln!("=== Z-Image Server ===");
    eprintln!("Backend:    {}", backend::BACKEND_NAME);
    eprintln!("Model dir:  {:?}", model_dir);
    eprintln!("Output dir: {:?}", output_dir);
    eprintln!("Port:       {}", port);
    eprintln!();

    // Shared state
    let tasks = Arc::new(RwLock::new(HashMap::new()));
    let models_loaded = Arc::new(AtomicBool::new(false));

    // Job channel
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

    // App state
    let state = Arc::new(AppState {
        tasks: tasks.clone(),
        job_sender: tx,
        models_loaded: models_loaded.clone(),
        backend_name: backend::BACKEND_NAME.to_string(),
        output_dir: output_dir.clone(),
        num_steps: Arc::new(AtomicUsize::new(8)),
        seed: Arc::new(AtomicU64::new(0)),
        use_seed: Arc::new(AtomicBool::new(false)),
        low_memory: Arc::new(AtomicBool::new(false)),
    });

    // Spawn GPU worker on dedicated thread
    queue::spawn_gpu_worker::<Backend>(
        rx,
        tasks,
        models_loaded,
        model_dir,
        output_dir,
        backend::create_device,
    );

    // Build router
    let app = Router::new()
        .route("/generate", post(handlers::generate))
        .route("/tasks/{id}", get(handlers::get_task))
        .route("/images/{id}", get(handlers::get_image))
        .route("/status", get(handlers::status))
        .route("/settings", post(handlers::update_settings))
        .layer(CorsLayer::permissive())
        .with_state(state);

    // Start server
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    eprintln!("Listening on http://{}", addr);
    eprintln!("Models are loading in the background...");
    eprintln!();

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
