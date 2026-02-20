# Z-Image App

**Pure Rust AI image generation** using the [Burn](https://burn.dev) deep learning framework.

A native implementation of Z-Image Turbo (a fast diffusion model) entirely in Rust, with optional native UI wrappers for macOS (SwiftUI) and cross-platform (egui).

## Project Vision

This project explores **what's possible with pure Rust ML inference**. The goal is not to compete with Python today, but to build on Rust's long-term advantages:

| Rust Advantage | Current State | Future Potential |
|----------------|---------------|------------------|
| Single binary distribution | Works | No Python env, just download and run |
| Memory safety | Works | Predictable behavior, no GC pauses |
| Cross-platform | Works | Same codebase for macOS/Windows/Linux |
| Performance | Similar to Python | Will improve as Burn/Metal mature |
| VRAM efficiency | Similar to Python | Flash attention coming to Burn |

### Honest Assessment

The Rust ML ecosystem (Burn, Candle, CubeCL) is young but growing fast. Currently:
- No significant speed advantage over Python (PyTorch/MLX)
- No significant VRAM advantage (naive attention, not flash attention yet)
- Fewer features than ComfyUI/diffusers

**But**: Single-binary deployment, true cross-compilation, and memory safety make this a compelling long-term bet as the ecosystem matures.

## Architecture

**Core Principle: Rust First** - All ML logic lives in Rust. UI layers are thin wrappers.

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interfaces                          │
├─────────────────────────┬───────────────────────────────────────┤
│   macOS Native (Swift)  │     Cross-Platform (egui/Rust)        │
│   - SwiftUI             │     - Windows, Linux, macOS           │
│   - Native look & feel  │     - Pure Rust, single binary        │
│   - UI only, no ML code │     - ~2000 lines                     │
└───────────┬─────────────┴──────────────┬────────────────────────┘
            │         C FFI Bridge        │
            ▼                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Rust Core (libz_image_ffi)                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ Image Gen   │  │ Text Chat   │  │ Memory Management       │  │
│  │ - Z-Image   │  │ - Qwen3     │  │ - Model caching         │  │
│  │ - Turbo     │  │ - 0.6B      │  │ - Attention slicing     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Burn Framework                          │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            │
│  │ Candle  │  │  WGPU   │  │  CUDA   │  │ ndarray │            │
│  │ +Metal  │  │ +Vulkan │  │(NVIDIA) │  │  (CPU)  │            │
│  │ (macOS) │  │         │  │         │  │         │            │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

| Component | Language | Purpose | Lines |
|-----------|----------|---------|-------|
| **Core Library** | Rust | All ML inference, model loading, memory mgmt | ~750 |
| **egui GUI** | Rust | Cross-platform native GUI | ~2000 |
| **SwiftUI App** | Swift | macOS-native look & feel (UI only) | ~1500 |
| **FFI Bridge** | Swift | Calls Rust functions | ~200 |

The Swift code contains **zero ML logic** - it only calls Rust FFI functions and renders UI.

## Required Repositories

This project uses relative path dependencies. Clone these repositories as siblings:

```
parent-directory/
├── burn/              # Burn framework (forked)
├── z-image-burn/      # Z-Image model implementation
├── qwen3-burn/        # Qwen3 model implementation
└── z-image-app/       # This repository
```

### Clone Commands

```bash
# Create parent directory
mkdir z-image-project && cd z-image-project

# Clone required repositories
git clone https://github.com/holg/burn.git
git clone https://github.com/holg/z-image-burn.git
git clone https://github.com/holg/qwen3-burn.git
git clone https://github.com/holg/z-image-app.git

# Build
cd z-image-app
cargo build --release --features "egui,metal"  # macOS
```

## Quick Start

### Cross-Platform egui GUI (Recommended)

#### macOS (Metal GPU)
```bash
cargo build --release --features "egui,metal"
./target/release/z-image-gui
```

#### Windows/Linux with NVIDIA GPU (CUDA)
```bash
cargo build --release --features "egui,cuda"
./target/release/z-image-gui      # Linux
.\target\release\z-image-gui.exe  # Windows
```

#### Windows/Linux with AMD/Intel GPU (Vulkan)
```bash
cargo build --release --features "egui,vulkan"
./target/release/z-image-gui
```

#### CPU only (any platform, slow)
```bash
cargo build --release --features "egui,cpu"
./target/release/z-image-gui
```

### Native macOS App (SwiftUI)

For native macOS look and feel:

```bash
# Default: Candle + Metal backend
./build_app.sh

# Alternative backends
./build_app.sh wgpu-metal    # WGPU + Metal (experimental)
./build_app.sh cpu           # CPU only (slow)

# Run
open ZImage.app
```

### Command Line Tools

```bash
# Generate an image
cargo run --release --bin test_generate --features metal -- \
    --model-dir ~/z-image-models \
    --prompt "A serene mountain landscape at sunset" \
    --output output.png \
    --width 512 --height 512 --steps 8

# Text generation
cargo run --release --bin test_chat --features metal -- \
    --model-dir ~/z-image-models/qwen3-0.6b \
    --prompt "Explain quantum computing"
```

## Features

- **Image Generation**: Z-Image Turbo diffusion model (4-20 steps, 256-1024px)
- **Text Chat**: Qwen3-0.6B conversational AI
- **Cross-platform**: macOS, Windows, Linux from same codebase
- **Multiple backends**: Metal, CUDA, Vulkan, CPU
- **Memory optimization**: Attention slicing, low-memory mode
- **Generation history**: Browse and reuse previous prompts

## Models

Download models from HuggingFace (or use the in-app downloader):

### Image Generation (~20 GB)
From [holgt/z-image-burn](https://huggingface.co/holgt/z-image-burn):
- `z_image_turbo_bf16.bpk` (12.3 GB) - Transformer
- `qwen3_4b_text_encoder.bpk` (8.0 GB) - Text encoder
- `ae.bpk` (198 MB) - Autoencoder
- `qwen3-tokenizer.json` (11 MB) - Tokenizer

### Text Chat (~1.5 GB)
From [holgt/qwen3-0.6b-burn](https://huggingface.co/holgt/qwen3-0.6b-burn):
- `model.bpk` (1.5 GB) - Qwen3-0.6B weights
- `tokenizer.json` (11 MB) - Tokenizer

## Memory Requirements

| Resolution | VRAM Required | Recommended Settings |
|------------|---------------|---------------------|
| 256x256    | ~12 GB        | Default |
| 512x512    | ~20 GB        | Default |
| 768x768    | ~24 GB        | Attention slicing: 4 |
| 1024x1024  | ~32 GB        | Attention slicing: 8, Low memory mode |

### Low VRAM Tips
- **Attention Slicing**: Set to 4-8 in Settings (trades speed for memory)
- **Low Memory Mode**: Unloads text encoder during diffusion (~7.5GB savings)
- **Smaller images**: Start with 256x256 or 512x512

## Project Structure

```
z-image-app/
├── src/
│   ├── lib.rs                 # FFI library (C-compatible API)
│   └── bin/
│       ├── gui.rs             # Cross-platform egui GUI
│       ├── gui_lowmem.rs      # Low-memory variant
│       ├── test_generate.rs   # CLI image generation
│       ├── test_chat.rs       # CLI text generation
│       ├── compute_embedding.rs
│       ├── generate_from_embedding.rs
│       └── convert_models.rs
├── ZImageApp/                 # macOS SwiftUI (UI only, no ML code)
│   ├── ZImageApp.swift
│   ├── ContentView.swift
│   └── ZImageBridge.swift     # FFI wrapper
├── z_image_ffi.h              # C header for FFI
├── build_app.sh               # macOS app builder
├── Cargo.toml
└── cubecl.toml                # GPU tuning config
```

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
    const char* prompt,
    const char* output_path,
    const char* model_dir,
    int32_t width,
    int32_t height
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

## Known Limitations & Future Work

### Current Limitations
- **Performance**: Similar to Python (Burn's Candle backend uses naive attention)
- **VRAM**: No significant advantage over PyTorch/MLX
- **Ecosystem**: Fewer models, less community support than Python

### Upstream Dependencies (Early/Unstable)
- **Burn Framework**: Pre-1.0, API may change
- **Candle Metal**: Missing optimized SDPA/Flash Attention
- **CubeCL**: Flash attention exists but WGPU-Metal integration experimental

### Planned Improvements
- [ ] Flash attention when Burn/Candle expose it
- [ ] INT8/INT4 quantization
- [ ] LoRA support
- [ ] Img2Img / Inpainting
- [ ] ControlNet

## Backend Requirements

### Metal (macOS)
- Apple Silicon or Intel Mac with Metal support
- No additional setup required

### CUDA (NVIDIA)
- NVIDIA GPU with CUDA support
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) installed
- Verify: `nvcc --version` and `nvidia-smi`

### Vulkan
- Vulkan-capable GPU (most modern GPUs)
- Vulkan runtime (usually included with GPU drivers)

## Troubleshooting

### High VRAM Usage
1. Enable low memory mode in Settings
2. Use smaller resolution (512x512)
3. Reduce attention slice size (try 4 or 2)
4. Use fewer inference steps (4-6)

### Slow Generation
1. Ensure GPU backend is active (not CPU)
2. Check model loading completed
3. Disable attention slicing (set to 0) if you have enough VRAM

### Black/Corrupted Output
1. Check model files are complete (not truncated downloads)
2. Verify model directory path is correct
3. Check console/logs for error messages

## Contributing

This is a research/exploration project. Contributions welcome for:
- Performance optimizations
- Additional backend support
- Model implementations
- Documentation

## License

Apache 2.0

## Acknowledgments

- [Burn](https://burn.dev) - Deep learning framework for Rust
- [Candle](https://github.com/huggingface/candle) - Minimalist ML framework
- [egui](https://github.com/emilk/egui) - Immediate mode GUI for Rust
- [Qwen](https://github.com/QwenLM/Qwen) - Language model
