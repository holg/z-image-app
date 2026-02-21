# Z-Image App

**Pure Rust AI image generation and fine-tuning** using the [Burn](https://burn.dev) deep learning framework.

A native implementation of [Z-Image Turbo](https://github.com/Tongyi-MAI/Z-Image) (a rectified flow model based on the S3-DiT architecture) entirely in Rust, with LoRA fine-tuning support and optional native UI wrappers for macOS (SwiftUI) and cross-platform (egui).

## Features

- **Image Generation**: Z-Image Turbo rectified flow model (4-20 steps, 256-1024px)
- **LoRA Fine-Tuning**: Train custom subjects (e.g., specific vehicles, characters) directly in Rust/Burn
- **Text Chat**: Qwen3-0.6B conversational AI
- **Cross-platform**: macOS, Windows, Linux from the same codebase
- **Multiple GPU backends**: Metal (macOS), CUDA (NVIDIA), Vulkan, CPU fallback
- **Memory optimization**: Attention slicing, sequence chunking, low-memory mode
- **Two UIs**: Native macOS SwiftUI app and cross-platform egui GUI

## Prerequisites

- **Rust** (stable, 2024 edition): https://rustup.rs
- **Git**
- **GPU**: Apple Silicon Mac (Metal), NVIDIA GPU (CUDA), or Vulkan-capable GPU
- **macOS SwiftUI build** (optional): Xcode command line tools (`xcode-select --install`)

## Required Repositories

This project uses **relative path dependencies** via Cargo. All repositories must be cloned as siblings in the same parent directory.

```
z-image-workspace/
├── burn/              # Burn deep learning framework (fork with fixes)
├── z-image-burn/      # Z-Image model implementation (FLUX VAE, S3-DiT transformer)
├── qwen3-burn/        # Qwen3 language model (text encoder + chat)
├── z-image-app/       # This repository (application, training, UI)
├── longcat-burn/      # (Optional) LongCat video generation model
└── umt5-burn/         # (Optional) UMT5 text encoder (for video generation)
```

### Setup

```bash
# Create workspace directory
mkdir z-image-workspace && cd z-image-workspace

# 1. Burn framework (fork with Z-Image compatibility fixes)
git clone https://github.com/holg/burn.git

# 2. Z-Image model (transformer, FLUX VAE, LoRA modules)
#    IMPORTANT: use the 'develop' branch
git clone -b develop https://github.com/holg/z-image-burn.git

# 3. Qwen3 language model (text encoding + chat)
git clone https://github.com/holg/qwen3-burn.git

# 4. This application
git clone https://github.com/holg/z-image-app.git

# Optional: for video generation (LongCat)
git clone https://github.com/holg/longcat-burn.git
git clone https://github.com/holg/umt5-burn.git
```

### Verify Setup

```bash
cd z-image-app

# Check all dependencies are reachable
ls ../burn/crates/burn/Cargo.toml      # Burn framework
ls ../z-image-burn/z-image/Cargo.toml  # Z-Image model
ls ../qwen3-burn/Cargo.toml            # Qwen3
# Optional (only needed with "video" feature):
# ls ../longcat-burn/Cargo.toml        # LongCat
# ls ../umt5-burn/Cargo.toml           # UMT5

# Build (macOS with Metal)
cargo build --release --features "egui,metal"
```

If any `ls` command fails, you're missing a repository or it's in the wrong location.

### Repository Overview

| Repository | Description | GitHub |
|------------|-------------|--------|
| [burn](https://github.com/holg/burn) | Fork of [Tracel AI's Burn](https://github.com/tracel-ai/burn) with compatibility fixes | `holg/burn` |
| [z-image-burn](https://github.com/holg/z-image-burn) | Z-Image S3-DiT transformer, FLUX VAE encoder/decoder, LoRA modules | `holg/z-image-burn` (branch: **develop**) |
| [qwen3-burn](https://github.com/holg/qwen3-burn) | Qwen3 model family: 4B text encoder for Z-Image + 0.6B for chat | `holg/qwen3-burn` |
| [longcat-burn](https://github.com/holg/longcat-burn) | LongCat 13.6B DiT for text/image-to-video generation (optional) | `holg/longcat-burn` |
| [umt5-burn](https://github.com/holg/umt5-burn) | UMT5 text encoder used by LongCat video generation (optional) | `holg/umt5-burn` |
| [z-image-app](https://github.com/holg/z-image-app) | This repo: application layer, training pipeline, GUIs | `holg/z-image-app` |

## Model Weights

Download pre-converted BurnPack (`.bpk`) model weights from HuggingFace. The app also includes an in-app downloader.

### Image Generation (~20 GB)

From [holgt/z-image-burn](https://huggingface.co/holgt/z-image-burn):

| File | Size | Description |
|------|------|-------------|
| `z_image_turbo_bf16.bpk` | 12.3 GB | Z-Image Turbo transformer (S3-DiT, 30 layers, dim 3840) |
| `qwen3_4b_text_encoder.bpk` | 8.0 GB | Qwen3-4B text encoder for prompt conditioning |
| `ae.bpk` | 198 MB | FLUX VAE autoencoder (encoder + decoder) |
| `qwen3-tokenizer.json` | 11 MB | Tokenizer for text encoding |

### Text Chat (~1.5 GB)

From [holgt/qwen3-0.6b-burn](https://huggingface.co/holgt/qwen3-0.6b-burn):

| File | Size | Description |
|------|------|-------------|
| `model.bpk` | 1.5 GB | Qwen3-0.6B causal language model |
| `tokenizer.json` | 11 MB | Tokenizer |

Place all image generation model files in a single directory (e.g., `~/z-image-models/`).

## Quick Start

### Cross-Platform egui GUI (Recommended)

```bash
# macOS (Metal GPU)
cargo run --release --bin z-image-gui --features "egui,metal"

# Windows/Linux with NVIDIA GPU (CUDA)
cargo run --release --bin z-image-gui --features "egui,cuda"

# Windows/Linux with AMD/Intel GPU (Vulkan)
cargo run --release --bin z-image-gui --features "egui,vulkan"

# CPU only (any platform, slow)
cargo run --release --bin z-image-gui --features "egui,cpu"
```

### Native macOS App (SwiftUI)

```bash
# Build and package as ZImage.app
./build_app.sh            # Default: Metal backend
./build_app.sh wgpu-metal # Alternative: WGPU + Metal
./build_app.sh cpu        # CPU only (slow)

# Run
open ZImage.app
```

### Command Line

```bash
# Generate an image
cargo run --release --bin test_generate --features metal -- \
    --model-dir ~/z-image-models \
    --prompt "A serene mountain landscape at sunset" \
    --output output.png \
    --width 512 --height 512

# Text chat
cargo run --release --bin test_chat --features metal -- \
    --model-dir ~/z-image-models \
    --prompt "Explain quantum computing"
```

## LoRA Fine-Tuning

Train the Z-Image model on custom subjects using LoRA (Low-Rank Adaptation) - entirely in Rust, no Python required.

### How It Works

1. **Base model stays frozen** - only small LoRA adapter matrices are trained
2. **Text encoder is NOT modified** - it provides prompt embeddings as-is
3. **Flow matching objective** - the training learns velocity prediction: `loss = MSE(v_predicted, v_target)`
4. **Result**: A small LoRA weights file (~50-100 MB) that customizes the base model

### Training Data Format

Prepare a directory with image files and matching `.txt` caption files:

```
training_data/uaz_469/
├── uaz_001.jpg
├── uaz_001.txt    → "a photo of sks uaz469 off-road vehicle on a mountain road"
├── uaz_002.jpg
├── uaz_002.txt    → "a sks uaz469 parked in front of a brick building, side view"
├── uaz_003.jpg
├── uaz_003.txt    → "a sks uaz469 driving through mud in the forest"
└── ...
```

Use a **rare trigger token** (like `sks`) + class noun (like `uaz469`) consistently across all captions. Vary the scene descriptions to teach the model the subject identity across different contexts.

Recommended: **10-20 images** per subject, different angles and backgrounds.

### Training Features

- **LoRA rank/alpha**: Configurable (default: rank 16, alpha 16.0)
- **Target layers**: Attention (qkv, to_out), Feed-Forward (w1, w2, w3), Refiners
- **Latent caching**: Pre-compute image latents and text embeddings to free VRAM during training
- **Training tab**: Integrated in the egui GUI with dataset browser, loss curve, and export

### Building with Training Support

```bash
cargo run --release --bin z-image-gui --features "egui,metal,train"
```

The `train` feature enables `burn/autodiff` for gradient computation.

## Architecture

**Core Principle: Rust First** - All ML logic (inference, training, model loading) lives in Rust. UI layers are thin wrappers.

```
┌──────────────────────────────────────────────────────────────────┐
│                        User Interfaces                           │
├──────────────────────────┬───────────────────────────────────────┤
│   macOS Native (Swift)   │     Cross-Platform (egui/Rust)        │
│   - SwiftUI              │     - Windows, Linux, macOS           │
│   - Native look & feel   │     - Pure Rust, single binary        │
│   - UI only, no ML code  │     - Includes Training tab           │
└────────────┬─────────────┴──────────────┬────────────────────────┘
             │         C FFI Bridge       │
             ▼                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Rust Core (z-image-app)                        │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────────┐  │
│  │ Image Gen   │  │ Text Chat   │  │ LoRA Training            │  │
│  │ - Z-Image   │  │ - Qwen3     │  │ - Flow matching          │  │
│  │ - Turbo     │  │ - 0.6B      │  │ - Latent caching         │  │
│  └─────────────┘  └─────────────┘  └──────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌──────────────────┐ ┌───────────────┐ ┌───────────────┐
│   z-image-burn   │ │  qwen3-burn   │ │ longcat-burn  │
│ - S3-DiT (30 L)  │ │ - 4B encoder  │ │ - 13.6B DiT   │
│ - FLUX VAE       │ │ - 0.6B chat   │ │ - Video gen   │
│ - LoRA modules   │ │ - Tokenizer   │ │ - UMT5 text   │
└──────────────────┘ └───────────────┘ └───────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Burn Framework (fork)                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │ Candle   │  │  WGPU    │  │  CUDA    │  │ ndarray  │         │
│  │ +Metal   │  │ +Vulkan  │  │ (NVIDIA) │  │  (CPU)   │         │
│  │ (macOS)  │  │          │  │          │  │          │         │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘         │
└──────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
z-image-app/
├── src/
│   ├── lib.rs                    # FFI library (C-compatible API)
│   ├── training/                 # LoRA training pipeline
│   │   ├── mod.rs
│   │   ├── dataset.rs            # Image + caption dataset loading
│   │   ├── latent_cache.rs       # Pre-computed latent/embedding cache
│   │   └── train_loop.rs         # Flow matching training loop
│   └── bin/
│       ├── gui.rs                # Cross-platform egui GUI (with Training tab)
│       ├── gui_lowmem.rs         # Low-memory GUI variant
│       ├── test_generate.rs      # CLI image generation
│       ├── test_chat.rs          # CLI text chat
│       ├── compute_embedding.rs  # Pre-compute text embeddings
│       ├── generate_from_embedding.rs
│       └── convert_models.rs     # Model format conversion
├── ZImageApp/                    # macOS SwiftUI (UI only, no ML code)
│   ├── ZImageApp.swift
│   ├── ContentView.swift
│   ├── ZImageBridge.swift        # FFI wrapper
│   ├── LongCatBridge.swift       # Video generation FFI wrapper
│   └── VideoGenerationView.swift
├── z_image_ffi.h                 # Generated C header for FFI
├── build_app.sh                  # macOS .app builder script
├── Cargo.toml
└── cubecl.toml                   # GPU tuning config
```

## Cargo Features

| Feature | Description |
|---------|-------------|
| `metal` | Metal GPU backend via Candle (macOS) |
| `cuda` | CUDA GPU backend via Candle (NVIDIA) |
| `vulkan` | Vulkan GPU backend via WGPU |
| `wgpu-metal` | Metal via WGPU (experimental) |
| `cpu` | CPU-only via ndarray (slow, any platform) |
| `egui` | Cross-platform GUI (eframe, egui_extras, rfd, etc.) |
| `train` | LoRA training support (burn/autodiff) |
| `video` | LongCat video generation (requires longcat-burn + umt5-burn) |
| `video-metal` | Video + Metal backend |
| `video-cuda` | Video + CUDA backend |

## Memory Requirements

| Resolution | VRAM Required | Recommended Settings |
|------------|---------------|---------------------|
| 256x256    | ~12 GB        | Default |
| 512x512    | ~20 GB        | Default |
| 768x768    | ~24 GB        | Attention slicing: 4 |
| 1024x1024  | ~32 GB        | Attention slicing: 8, Low memory mode |

### Low VRAM Tips

- **Low Memory Mode**: Unloads text encoder during generation (~7.5 GB savings)
- **Attention Slicing**: Set to 4-8 in Settings (trades speed for memory)
- **Sequence Chunking**: Reduces peak memory for attention computation
- **Smaller images**: Start with 256x256 or 512x512

## FFI API

The Rust library exposes a C-compatible API for integration with any language:

```c
// Initialize GPU device
int32_t z_image_init(void);

// Model lifecycle
int32_t z_image_load_models(const char* model_dir);
int32_t z_image_unload_models(void);
int32_t z_image_models_loaded(void);

// Image generation
int32_t z_image_generate(
    const char* prompt, const char* output_path,
    const char* model_dir, int32_t width, int32_t height
);

// Memory optimization
int32_t z_image_set_attention_slice_size(int32_t size);
int32_t z_image_set_low_memory_mode(int32_t enabled);
int32_t z_image_set_num_steps(int32_t steps);
int32_t z_image_set_seed(uint64_t seed);

// Text generation (Qwen3)
int32_t qwen3_init(const char* model_dir);
char* qwen3_generate(const char* prompt, int32_t max_tokens, float temperature);
void qwen3_free_string(char* str);
int32_t qwen3_unload(void);
```

## Known Limitations

- **Performance**: Similar to Python (Burn's Candle backend uses naive attention, no Flash Attention yet)
- **VRAM**: No significant advantage over PyTorch/MLX currently
- **Ecosystem**: Younger than Python ML ecosystem
- **Burn Framework**: Pre-1.0, API may change

## Roadmap

- [x] Z-Image Turbo inference
- [x] Multiple GPU backends (Metal, CUDA, Vulkan)
- [x] Attention slicing and low-memory mode
- [x] LoRA fine-tuning pipeline
- [x] FLUX VAE encoder (for training)
- [x] egui Training tab with loss curve
- [ ] LoRA weight save/load (safetensors)
- [ ] LoRA merge and inference
- [ ] Flash attention (when Burn/Candle expose it)
- [ ] INT8/INT4 quantization
- [ ] Img2Img / Inpainting

## Contributing

Contributions welcome! To get started:

1. Clone all [required repositories](#required-repositories) as described above
2. Run the [verify setup](#verify-setup) steps
3. Build with `cargo build --release --features "egui,metal"` (or your GPU backend)
4. Check existing issues or open a new one to discuss your contribution

Areas where help is appreciated:
- Performance optimizations (attention, memory)
- Additional backend testing (CUDA, Vulkan)
- Training improvements
- Documentation

## License

Apache 2.0

## Acknowledgments

- [Z-Image / S3-DiT](https://github.com/Tongyi-MAI/Z-Image) by Tongyi-MAI (Alibaba) - The original rectified flow model
- [Burn](https://burn.dev) by Tracel AI - Deep learning framework for Rust
- [Candle](https://github.com/huggingface/candle) by Hugging Face - ML framework powering the Metal/CUDA backends
- [egui](https://github.com/emilk/egui) - Immediate mode GUI for Rust
- [Qwen3](https://github.com/QwenLM/Qwen3) by Alibaba - Language model family
- [FLUX](https://github.com/black-forest-labs/flux) by Black Forest Labs - VAE autoencoder architecture
