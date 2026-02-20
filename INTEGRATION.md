# ZImage + LongCat Video Integration Guide

This guide explains how to integrate the LongCat video generation library with the ZImage SwiftUI app.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     ZImageApp (SwiftUI)                         │
├────────┬────────┬────────┬────────┬─────────┬──────────────────┤
│ Image  │ Video  │History │  Chat  │Settings │      Logs        │
│(ZImage)│(LongCat)│       │ (Qwen3)│         │                  │
└───┬────┴───┬────┴────────┴────────┴─────────┴──────────────────┘
    │        │
    ▼        ▼
┌────────┐  ┌────────────┐
│ZImage  │  │ LongCat    │
│Bridge  │  │ Bridge     │
│(Swift) │  │ (Swift)    │
└───┬────┘  └─────┬──────┘
    │             │
    ▼             ▼
┌────────┐  ┌────────────┐
│libz_   │  │liblongcat_ │
│image.a │  │ffi.a       │
│(Rust)  │  │(Rust)      │
└────────┘  └────────────┘
```

## Build Instructions

### 1. Build the Rust Libraries

```bash
# Build ZImage FFI (if not already built)
cd z-image-burn
cargo build --release --features metal

# Build LongCat FFI
cd ../longcat-burn/longcat-ffi
cargo build --release --features metal
```

### 2. Library Locations

After building, the libraries are at:

| Library | Path |
|---------|------|
| ZImage | `z-image-burn/target/release/libz_image.a` |
| LongCat | `longcat-burn/longcat-ffi/target/release/liblongcat_ffi.a` |

### 3. Header Files

| Header | Path |
|--------|------|
| LongCat | `longcat-burn/longcat-ffi/include/longcat.h` |

### 4. Xcode Project Setup

#### Add Libraries to Project

1. Open `ZImageApp.xcodeproj` in Xcode
2. Select the project in the navigator
3. Select the `ZImageApp` target
4. Go to **Build Phases** → **Link Binary With Libraries**
5. Click **+** and add:
   - `liblongcat_ffi.a` (drag from Finder)
   - System frameworks if needed (Metal, Accelerate)

#### Add Header Search Paths

1. Go to **Build Settings**
2. Find **Header Search Paths**
3. Add: `$(PROJECT_DIR)/../longcat-burn/longcat-ffi/include`

#### Add Library Search Paths

1. Go to **Build Settings**
2. Find **Library Search Paths**
3. Add: `$(PROJECT_DIR)/../longcat-burn/longcat-ffi/target/release`

#### Swift Files

The following Swift files need to be in the project:

- `ZImageBridge.swift` - Bridge to ZImage Rust library
- `LongCatBridge.swift` - Bridge to LongCat Rust library (NEW)
- `VideoGenerationView.swift` - Video generation UI (NEW)
- `ContentView.swift` - Updated with Video tab

### 5. Model Files

LongCat requires these model files in the models directory:

| File | Description | Size |
|------|-------------|------|
| `longcat_dit.safetensors` or `.bpk` | DiT transformer (13.6B params) | ~27GB |
| `wan_vae.safetensors` or `.bpk` | WAN 2.1 Video VAE | ~500MB |
| `umt5_xxl.safetensors` or `.bpk` | UMT5-XXL text encoder | ~10GB |

## Video Generation Workflow

### Text-to-Video (T2V)

1. User enters a text prompt
2. ZImage generates the first frame
3. LongCat generates video from first frame + prompt
4. Video is displayed/saved

### Image-to-Video (I2V)

1. User provides an image (or generates one with ZImage)
2. User enters a motion/action prompt
3. LongCat generates video from image + prompt
4. Video is displayed/saved

## Memory Requirements

| Resolution | Frames | VRAM Required |
|------------|--------|---------------|
| 480p | 41 (2.7s) | ~24GB |
| 480p | 81 (5.4s) | ~32GB |
| 720p | 81 (2.7s) | ~48GB |

## Troubleshooting

### "Models not loaded" error
- Check that model files exist in the specified directory
- Ensure file names match expected names
- Check Logs tab for detailed error messages

### Out of memory
- Use 480p resolution instead of 720p
- Reduce number of frames
- Reduce inference steps
- Enable attention slicing in ZImage settings

### Slow generation
- Ensure release build is used (not debug)
- Check that Metal GPU is being used
- Monitor Activity Monitor for GPU usage

## API Reference

### LongCatBridge

```swift
// Initialize
LongCatBridge.shared.initialize()

// Load models
let success = LongCatBridge.shared.loadModels(modelDir: "/path/to/models")

// Generate video
let success = LongCatBridge.shared.generateVideoFromImage(
    imagePath: "/path/to/image.png",
    prompt: "The cat walks across the room",
    outputPath: "/path/to/output.mp4"
)

// Settings
LongCatBridge.shared.setNumSteps(50)
LongCatBridge.shared.setGuidanceScale(5.0)
LongCatBridge.shared.setNumFrames(81)
LongCatBridge.shared.setVideoSize(width: 832, height: 480)
LongCatBridge.shared.setVideoFPS(15)
```

### VideoPreset

```swift
// Apply a preset
VideoPreset.fast480p.apply()    // 480p, 2s, 25 steps
VideoPreset.standard480p.apply() // 480p, 5s, 50 steps
VideoPreset.quality720p.apply()  // 720p, 5s, 50 steps
```
