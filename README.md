# Z-Image App

Cross-platform image generation and text chat application using the Burn deep learning framework.

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

## Quick Start - egui GUI

### macOS (Metal GPU)
```bash
cargo build --release --features "egui,metal"
./target/release/z-image-gui
```

### Windows/Linux with NVIDIA GPU (CUDA)
```bash
cargo build --release --features "egui,cuda"
./target/release/z-image-gui      # Linux
.\target\release\z-image-gui.exe  # Windows
```

### Windows/Linux with AMD/Intel GPU (Vulkan)
```bash
cargo build --release --features "egui,vulkan"
./target/release/z-image-gui      # Linux
.\target\release\z-image-gui.exe  # Windows
```

### CPU only (any platform, slow)
```bash
cargo build --release --features "egui,cpu"
./target/release/z-image-gui
```

## Features

- **Image Generation**: Z-Image Turbo diffusion model for text-to-image
- **Text Chat**: Qwen3-0.6B for conversational AI
- **Cross-platform**: macOS, Windows, Linux
- **Multiple backends**: Metal, CUDA, Vulkan, CPU

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
- **Attention Slicing**: Set to 4-8 in Settings
- **Low Memory Mode**: Unloads text encoder during diffusion
- **Smaller images**: Start with 256x256 or 512x512

## Backend Requirements

### CUDA (NVIDIA)
- NVIDIA GPU with CUDA support
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) installed
- Verify: `nvcc --version` and `nvidia-smi`

### Vulkan
- Vulkan-capable GPU (most modern GPUs)
- Vulkan runtime (usually included with GPU drivers)

### Metal (macOS)
- Apple Silicon or Intel Mac with Metal support
- No additional setup required

## Native macOS App (SwiftUI)

For the native macOS app with SwiftUI:

```bash
./build_app.sh
```

This builds a `.app` bundle using Metal acceleration.

## FFI Library

The library can be used from other languages via FFI:

```bash
cargo build --release
```

Produces `libz_image_ffi.dylib` (macOS) / `.so` (Linux) / `.dll` (Windows)

Header file: `z_image_ffi.h`

## License

Apache 2.0
