//! LongCat Video Generation FFI API
//!
//! Feature-gated behind `video`. Provides C-compatible interface for video generation.
//!
//! Three modes of operation:
//!
//! 1. TEXT-TO-VIDEO (T2V) - Main LongCat purpose
//!    - longcat_init() -> longcat_load_models() -> longcat_generate_video()
//!    - Only loads LongCat models (DiT + VAE + UMT5)
//!
//! 2. IMAGE-TO-VIDEO (I2V) - Animate existing image
//!    - longcat_init() -> longcat_load_models() -> longcat_generate_video_from_image()
//!    - User provides image path + text description
//!
//! 3. TEXT-TO-IMAGE-TO-VIDEO (T2I2V) - Full pipeline
//!    - z_image_init() -> z_image_load_models() + longcat_load_models()
//!    - -> longcat_generate_video_from_text_with_zimage()
//!    - Requires both z-image AND longcat models loaded

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::Instant;

use burn::backend::candle::{Candle, CandleDevice};
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use half::bf16;
use once_cell::sync::OnceCell as OnceCell2;
use parking_lot::Mutex as ParkingMutex;
use longcat_burn::{GenerateConfig, LongCatPipeline, MemoryConfig, PipelineBuilder, set_attention_slice_size};
use longcat_burn::{WanVae, WanVaeConfig, GenerationControl};

use crate::{Backend, CpuBackend, DEVICE, IMAGE_MODELS, z_image_models_loaded, z_image_free_string};
use z_image::GenerateFromTextOpts;

// LongCat global state
static LONGCAT_PIPELINE: OnceCell2<ParkingMutex<Option<LongCatPipeline<Backend>>>> = OnceCell2::new();
static LONGCAT_MODELS_LOADED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

// Generation control for pause/cancel
static LONGCAT_CONTROL: OnceCell2<GenerationControl> = OnceCell2::new();

// CPU VAE for I2V encoding (Candle Metal doesn't support 3D conv)
static LONGCAT_CPU_VAE: OnceCell2<ParkingMutex<Option<WanVae<CpuBackend>>>> = OnceCell2::new();
static LONGCAT_VAE_PATH: OnceLock<PathBuf> = OnceLock::new();

// LongCat generation settings
static LONGCAT_NUM_STEPS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(50);
static LONGCAT_GUIDANCE_SCALE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(500); // 5.0 * 100
static LONGCAT_NUM_FRAMES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(81);
static LONGCAT_VIDEO_WIDTH: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(832);
static LONGCAT_VIDEO_HEIGHT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(480);
static LONGCAT_VIDEO_FPS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(15);

/// Initialize the LongCat GPU device.
/// Can be called independently of z_image_init() for pure T2V/I2V modes.
#[unsafe(no_mangle)]
pub extern "C" fn longcat_init() -> i32 {
    // Check if device already initialized (by z-image or previous longcat_init)
    if DEVICE.get().is_some() {
        eprintln!("[longcat] Using existing device");
        return 0;
    }

    // Initialize Metal device directly (no z-image dependency)
    eprintln!("[longcat] Initializing Metal device...");
    let device = CandleDevice::metal(0);
    match DEVICE.set(device) {
        Ok(_) => {
            eprintln!("[longcat] Metal device initialized");
            0
        }
        Err(_) => {
            eprintln!("[longcat] Device already initialized");
            0
        }
    }
}

/// Load LongCat video generation models
#[unsafe(no_mangle)]
pub extern "C" fn longcat_load_models(model_dir: *const c_char) -> i32 {
    let model_dir = unsafe {
        if model_dir.is_null() {
            eprintln!("[longcat] Error: model_dir is null");
            return -1;
        }
        match CStr::from_ptr(model_dir).to_str() {
            Ok(s) => PathBuf::from(s),
            Err(_) => {
                eprintln!("[longcat] Error: Invalid UTF-8 in model_dir");
                return -1;
            }
        }
    };

    let device = match DEVICE.get() {
        Some(d) => d,
        None => {
            eprintln!("[longcat] Error: Device not initialized. Call longcat_init first.");
            return -1;
        }
    };

    eprintln!("[longcat] Loading models from {:?}...", model_dir);

    // Build pipeline
    let mut builder = PipelineBuilder::new(device.clone());

    // Look for model files
    let dit_path = find_longcat_model(&model_dir, "longcat_dit");
    let vae_path = find_longcat_model(&model_dir, "wan_vae");
    let text_path = find_longcat_model(&model_dir, "umt5_xxl");

    if let Some(path) = dit_path {
        eprintln!("[longcat] Found DiT weights: {:?}", path);
        builder = builder.with_dit_weights(path);
    }

    if let Some(path) = vae_path {
        eprintln!("[longcat] Found VAE weights: {:?}", path);
        // Save path for CPU VAE loading later (for I2V mode)
        let _ = LONGCAT_VAE_PATH.set(path.clone());
        builder = builder.with_vae_weights(path);
    }

    if let Some(path) = text_path {
        eprintln!("[longcat] Found text encoder weights: {:?}", path);
        builder = builder.with_text_weights(path);
    }

    match builder.build() {
        Ok(pipeline) => {
            let pipeline_cell = LONGCAT_PIPELINE.get_or_init(|| ParkingMutex::new(None));
            *pipeline_cell.lock() = Some(pipeline);
            LONGCAT_MODELS_LOADED.store(true, std::sync::atomic::Ordering::SeqCst);
            eprintln!("[longcat] Models loaded successfully");
            0
        }
        Err(e) => {
            eprintln!("[longcat] Error: Failed to build pipeline: {}", e);
            -1
        }
    }
}

fn find_longcat_model(dir: &PathBuf, name: &str) -> Option<PathBuf> {
    for ext in &["safetensors", "bpk"] {
        let path = dir.join(format!("{}.{}", name, ext));
        if path.exists() {
            return Some(path);
        }
    }
    None
}

/// Unload LongCat models from memory
#[unsafe(no_mangle)]
pub extern "C" fn longcat_unload_models() -> i32 {
    if let Some(pipeline_cell) = LONGCAT_PIPELINE.get() {
        *pipeline_cell.lock() = None;
    }
    LONGCAT_MODELS_LOADED.store(false, std::sync::atomic::Ordering::SeqCst);
    eprintln!("[longcat] Models unloaded");
    0
}

/// Check if LongCat models are loaded
#[unsafe(no_mangle)]
pub extern "C" fn longcat_models_loaded() -> i32 {
    if LONGCAT_MODELS_LOADED.load(std::sync::atomic::Ordering::SeqCst) { 1 } else { 0 }
}

// ============================================================================
// Mode 1: TEXT-TO-VIDEO (T2V)
// ============================================================================

/// Generate video directly from text prompt (T2V mode).
#[unsafe(no_mangle)]
pub extern "C" fn longcat_generate_video(
    prompt: *const c_char,
    output_path: *const c_char,
) -> i32 {
    let prompt = unsafe {
        if prompt.is_null() { return -1; }
        match CStr::from_ptr(prompt).to_str() {
            Ok(s) => s.to_string(),
            Err(_) => return -1,
        }
    };

    let output_path = unsafe {
        if output_path.is_null() { return -1; }
        match CStr::from_ptr(output_path).to_str() {
            Ok(s) => PathBuf::from(s),
            Err(_) => return -1,
        }
    };

    let pipeline_cell = match LONGCAT_PIPELINE.get() {
        Some(p) => p,
        None => {
            eprintln!("[longcat] Error: Models not loaded");
            return -1;
        }
    };

    let pipeline_guard = pipeline_cell.lock();
    let pipeline = match pipeline_guard.as_ref() {
        Some(p) => p,
        None => {
            eprintln!("[longcat] Error: Models not loaded");
            return -1;
        }
    };

    let config = GenerateConfig {
        num_frames: LONGCAT_NUM_FRAMES.load(std::sync::atomic::Ordering::Relaxed),
        height: LONGCAT_VIDEO_HEIGHT.load(std::sync::atomic::Ordering::Relaxed),
        width: LONGCAT_VIDEO_WIDTH.load(std::sync::atomic::Ordering::Relaxed),
        fps: LONGCAT_VIDEO_FPS.load(std::sync::atomic::Ordering::Relaxed),
        num_inference_steps: LONGCAT_NUM_STEPS.load(std::sync::atomic::Ordering::Relaxed),
        guidance_scale: LONGCAT_GUIDANCE_SCALE.load(std::sync::atomic::Ordering::Relaxed) as f32 / 100.0,
        seed: None,
    };

    eprintln!("[longcat] Generating video: \"{}\"", prompt);
    eprintln!("[longcat] Config: {}x{} @ {}fps, {} frames, {} steps",
              config.width, config.height, config.fps, config.num_frames, config.num_inference_steps);

    // Get or create control handle
    let control = LONGCAT_CONTROL.get_or_init(GenerationControl::new);

    // Reset control state for new generation
    control.resume(); // Clear any pause state

    // Catch panics to prevent app crash
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pipeline.generate_with_control(&prompt, &config, control)
    }));

    match result {
        Ok(video) => {
            eprintln!("[longcat] Generated video shape: {:?}", video.dims());
            eprintln!("[longcat] Output path: {:?}", output_path);
            0
        }
        Err(e) => {
            let msg = if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else {
                "Unknown panic".to_string()
            };
            if msg.contains("cancelled") {
                eprintln!("[longcat] Generation was cancelled");
                return -3;
            }
            eprintln!("[longcat] Error: Generation failed - {}", msg);
            -1
        }
    }
}

// ============================================================================
// Mode 2: IMAGE-TO-VIDEO (I2V)
// ============================================================================

/// Generate video from an existing image (I2V mode).
///
/// NOTE: Currently disabled on Metal backend due to missing 3D convolution support.
#[unsafe(no_mangle)]
pub extern "C" fn longcat_generate_video_from_image(
    _image_path: *const c_char,
    _prompt: *const c_char,
    _output_path: *const c_char,
) -> i32 {
    eprintln!("[longcat] Error: I2V mode is not available on Metal backend");
    eprintln!("[longcat] Reason: VAE encoder requires 3D convolutions which Candle Metal doesn't support");
    eprintln!("[longcat] Workaround: Use T2V mode instead (Text-to-Video)");
    return -2;
}

/// Generate video from an existing image (I2V mode) - INTERNAL IMPLEMENTATION
#[allow(dead_code)]
fn longcat_generate_video_from_image_impl(
    image_path: *const c_char,
    prompt: *const c_char,
    output_path: *const c_char,
) -> i32 {
    let image_path = unsafe {
        if image_path.is_null() { return -1; }
        match CStr::from_ptr(image_path).to_str() {
            Ok(s) => PathBuf::from(s),
            Err(_) => return -1,
        }
    };

    let prompt = unsafe {
        if prompt.is_null() { return -1; }
        match CStr::from_ptr(prompt).to_str() {
            Ok(s) => s.to_string(),
            Err(_) => return -1,
        }
    };

    let output_path = unsafe {
        if output_path.is_null() { return -1; }
        match CStr::from_ptr(output_path).to_str() {
            Ok(s) => PathBuf::from(s),
            Err(_) => return -1,
        }
    };

    // Load image
    let image = match image::open(&image_path) {
        Ok(img) => img.to_rgb8(),
        Err(e) => {
            eprintln!("[longcat] Error loading image: {}", e);
            return -1;
        }
    };

    let (img_width, img_height) = image.dimensions();
    eprintln!("[longcat] I2V: Loaded image {}x{} from {:?}", img_width, img_height, image_path);
    eprintln!("[longcat] Prompt: \"{}\"", prompt);

    // Check if VAE path is available
    let vae_path = match LONGCAT_VAE_PATH.get() {
        Some(p) => p.clone(),
        None => {
            eprintln!("[longcat] Error: VAE path not set. Load models first.");
            return -1;
        }
    };

    let pipeline_cell = match LONGCAT_PIPELINE.get() {
        Some(p) => p,
        None => {
            eprintln!("[longcat] Error: Models not loaded");
            return -1;
        }
    };

    let config = GenerateConfig {
        num_frames: LONGCAT_NUM_FRAMES.load(std::sync::atomic::Ordering::Relaxed),
        height: LONGCAT_VIDEO_HEIGHT.load(std::sync::atomic::Ordering::Relaxed),
        width: LONGCAT_VIDEO_WIDTH.load(std::sync::atomic::Ordering::Relaxed),
        fps: LONGCAT_VIDEO_FPS.load(std::sync::atomic::Ordering::Relaxed),
        num_inference_steps: LONGCAT_NUM_STEPS.load(std::sync::atomic::Ordering::Relaxed),
        guidance_scale: LONGCAT_GUIDANCE_SCALE.load(std::sync::atomic::Ordering::Relaxed) as f32 / 100.0,
        seed: None,
    };

    let gpu_device = DEVICE.get().expect("Device not initialized");
    let cpu_device = NdArrayDevice::Cpu;

    // Convert image to tensor [1, 3, H, W] in range [0, 1]
    let img_data: Vec<f32> = image.pixels()
        .flat_map(|p| p.0.iter().map(|&v| v as f32 / 255.0))
        .collect();

    // Reshape from HWC to CHW
    let mut chw_data = vec![0.0f32; (3 * img_height * img_width) as usize];
    for y in 0..img_height as usize {
        for x in 0..img_width as usize {
            for c in 0..3 {
                chw_data[c * (img_height as usize * img_width as usize) + y * img_width as usize + x] =
                    img_data[(y * img_width as usize + x) * 3 + c];
            }
        }
    }

    eprintln!("[longcat] Generating I2V: {}x{} @ {}fps, {} frames, {} steps",
              config.width, config.height, config.fps, config.num_frames, config.num_inference_steps);

    // Step 1: Load CPU VAE if not already loaded
    eprintln!("[longcat] Step 1: Encoding image with CPU VAE...");
    let cpu_vae_cell = LONGCAT_CPU_VAE.get_or_init(|| ParkingMutex::new(None));
    {
        let mut cpu_vae_guard = cpu_vae_cell.lock();
        if cpu_vae_guard.is_none() {
            eprintln!("[longcat] Loading CPU VAE from {:?}...", vae_path);
            let vae_config = WanVaeConfig::default();
            let mut vae: WanVae<CpuBackend> = vae_config.init(&cpu_device);

            if let Err(e) = vae.load_weights(&vae_path) {
                eprintln!("[longcat] Error loading CPU VAE: {:?}", e);
                return -1;
            }

            *cpu_vae_guard = Some(vae);
            eprintln!("[longcat] CPU VAE loaded");
        }
    }

    // Step 2: Encode image on CPU
    let image_latent_data: Vec<f32> = {
        let cpu_vae_guard = cpu_vae_cell.lock();
        let cpu_vae = cpu_vae_guard.as_ref().unwrap();

        let image_cpu: burn::tensor::Tensor<CpuBackend, 4> = {
            let tensor_1d: burn::tensor::Tensor<CpuBackend, 1> =
                burn::tensor::Tensor::from_floats(chw_data.as_slice(), &cpu_device);
            tensor_1d.reshape([1, 3, img_height as usize, img_width as usize])
        };

        let image_video = image_cpu.reshape([1, 3, 1, img_height as usize, img_width as usize]);
        let latent = cpu_vae.encode_deterministic(image_video);
        eprintln!("[longcat] Encoded latent shape: {:?}", latent.dims());
        latent.into_data().to_vec().unwrap()
    };

    // Step 3: Transfer latent to GPU and run generation
    eprintln!("[longcat] Step 2: Transferring latent to GPU and generating video...");

    let pipeline_guard = pipeline_cell.lock();
    let pipeline = match pipeline_guard.as_ref() {
        Some(p) => p,
        None => {
            eprintln!("[longcat] Error: Models not loaded");
            return -1;
        }
    };

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let latent_gpu: burn::tensor::Tensor<Backend, 5> = {
            let latent_c = 16;
            let latent_t = 1;
            let latent_h = img_height as usize / 8;
            let latent_w = img_width as usize / 8;

            let tensor_1d: burn::tensor::Tensor<Backend, 1> =
                burn::tensor::Tensor::from_floats(image_latent_data.as_slice(), gpu_device);
            tensor_1d.reshape([1, latent_c, latent_t, latent_h, latent_w])
        };

        pipeline.generate_from_latent(latent_gpu, &prompt, &config)
    }));

    match result {
        Ok(video) => {
            eprintln!("[longcat] Generated video shape: {:?}", video.dims());
            eprintln!("[longcat] Output path: {:?}", output_path);
            0
        }
        Err(e) => {
            let msg = if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else {
                "Unknown panic".to_string()
            };
            eprintln!("[longcat] Error: I2V generation failed - {}", msg);
            -1
        }
    }
}

// ============================================================================
// Mode 3: TEXT-TO-IMAGE-TO-VIDEO (T2I2V)
// ============================================================================

/// Generate video from text by first generating an image with z-image (T2I2V mode).
#[unsafe(no_mangle)]
pub extern "C" fn longcat_generate_video_from_text_with_zimage(
    prompt: *const c_char,
    output_path: *const c_char,
) -> i32 {
    let prompt = unsafe {
        if prompt.is_null() { return -1; }
        match CStr::from_ptr(prompt).to_str() {
            Ok(s) => s.to_string(),
            Err(_) => return -1,
        }
    };

    let output_path = unsafe {
        if output_path.is_null() { return -1; }
        match CStr::from_ptr(output_path).to_str() {
            Ok(s) => PathBuf::from(s),
            Err(_) => return -1,
        }
    };

    // Check if z-image models are loaded
    if z_image_models_loaded() == 0 {
        eprintln!("[longcat] Error: z-image models not loaded. Call z_image_load_models() first.");
        return -1;
    }

    // Check if LongCat models are loaded
    if longcat_models_loaded() == 0 {
        eprintln!("[longcat] Error: LongCat models not loaded. Call longcat_load_models() first.");
        return -1;
    }

    eprintln!("[longcat] T2I2V mode: \"{}\"", prompt);

    let width = LONGCAT_VIDEO_WIDTH.load(std::sync::atomic::Ordering::Relaxed);
    let height = LONGCAT_VIDEO_HEIGHT.load(std::sync::atomic::Ordering::Relaxed);

    eprintln!("[longcat] Step 1: Generating initial frame with z-image ({}x{})...", width, height);

    let temp_image_path = output_path.with_extension("_temp_frame.png");

    let device = DEVICE.get().expect("Device not initialized");

    let models_mutex = IMAGE_MODELS.get().expect("Image models not initialized");
    let models_guard = models_mutex.lock().expect("Lock error");
    let zimage_models = models_guard.as_ref().expect("z-image models not loaded");

    let gen_start = Instant::now();
    let opts = GenerateFromTextOpts {
        prompt: prompt.clone(),
        out_path: temp_image_path.clone(),
        width,
        height,
        num_inference_steps: Some(8),
        seed: None,
    };

    if let Err(e) = z_image::generate_from_text(
        &opts,
        &zimage_models.tokenizer,
        &zimage_models.text_encoder,
        &zimage_models.autoencoder,
        &zimage_models.transformer,
        device,
    ) {
        eprintln!("[longcat] Error generating initial frame: {:?}", e);
        return -1;
    }

    eprintln!("[longcat] Initial frame generated in {:.2}s", gen_start.elapsed().as_secs_f32());

    drop(models_guard);

    eprintln!("[longcat] Step 2: Animating frame with LongCat...");

    let image = match image::open(&temp_image_path) {
        Ok(img) => img.to_rgb8(),
        Err(e) => {
            eprintln!("[longcat] Error loading generated frame: {}", e);
            return -1;
        }
    };

    let (img_width, img_height) = image.dimensions();

    let pipeline_cell = LONGCAT_PIPELINE.get().expect("Pipeline not initialized");
    let pipeline_guard = pipeline_cell.lock();
    let pipeline = pipeline_guard.as_ref().expect("LongCat models not loaded");

    let config = GenerateConfig {
        num_frames: LONGCAT_NUM_FRAMES.load(std::sync::atomic::Ordering::Relaxed),
        height: LONGCAT_VIDEO_HEIGHT.load(std::sync::atomic::Ordering::Relaxed),
        width: LONGCAT_VIDEO_WIDTH.load(std::sync::atomic::Ordering::Relaxed),
        fps: LONGCAT_VIDEO_FPS.load(std::sync::atomic::Ordering::Relaxed),
        num_inference_steps: LONGCAT_NUM_STEPS.load(std::sync::atomic::Ordering::Relaxed),
        guidance_scale: LONGCAT_GUIDANCE_SCALE.load(std::sync::atomic::Ordering::Relaxed) as f32 / 100.0,
        seed: None,
    };

    let img_data: Vec<f32> = image.pixels()
        .flat_map(|p| p.0.iter().map(|&v| v as f32 / 255.0))
        .collect();

    let mut chw_data = vec![0.0f32; (3 * img_height * img_width) as usize];
    for y in 0..img_height as usize {
        for x in 0..img_width as usize {
            for c in 0..3 {
                chw_data[c * (img_height as usize * img_width as usize) + y * img_width as usize + x] =
                    img_data[(y * img_width as usize + x) * 3 + c];
            }
        }
    }

    let image_tensor: burn::tensor::Tensor<Backend, 4> = {
        let tensor_1d: burn::tensor::Tensor<Backend, 1> =
            burn::tensor::Tensor::from_floats(chw_data.as_slice(), device);
        tensor_1d.reshape([1, 3, img_height as usize, img_width as usize])
    };

    let video_start = Instant::now();

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pipeline.generate_from_image(image_tensor, &prompt, &config)
    }));

    let _ = std::fs::remove_file(&temp_image_path);

    match result {
        Ok(video) => {
            eprintln!("[longcat] Video generated in {:.2}s", video_start.elapsed().as_secs_f32());
            eprintln!("[longcat] Video shape: {:?}", video.dims());
            eprintln!("[longcat] Output: {:?}", output_path);
            0
        }
        Err(e) => {
            let msg = if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else {
                "Unknown panic".to_string()
            };
            eprintln!("[longcat] Error: T2I2V generation failed - {}", msg);
            -1
        }
    }
}

/// Set number of inference steps for video generation
#[unsafe(no_mangle)]
pub extern "C" fn longcat_set_num_steps(steps: i32) {
    LONGCAT_NUM_STEPS.store(steps.max(1) as usize, std::sync::atomic::Ordering::Relaxed);
}

/// Get number of inference steps
#[unsafe(no_mangle)]
pub extern "C" fn longcat_get_num_steps() -> i32 {
    LONGCAT_NUM_STEPS.load(std::sync::atomic::Ordering::Relaxed) as i32
}

/// Set guidance scale (stored as integer: scale * 100)
#[unsafe(no_mangle)]
pub extern "C" fn longcat_set_guidance_scale(scale: f32) {
    LONGCAT_GUIDANCE_SCALE.store((scale.max(1.0) * 100.0) as u32, std::sync::atomic::Ordering::Relaxed);
}

/// Get guidance scale
#[unsafe(no_mangle)]
pub extern "C" fn longcat_get_guidance_scale() -> f32 {
    LONGCAT_GUIDANCE_SCALE.load(std::sync::atomic::Ordering::Relaxed) as f32 / 100.0
}

/// Set number of frames to generate
#[unsafe(no_mangle)]
pub extern "C" fn longcat_set_num_frames(frames: i32) {
    LONGCAT_NUM_FRAMES.store(frames.max(1) as usize, std::sync::atomic::Ordering::Relaxed);
}

/// Get number of frames
#[unsafe(no_mangle)]
pub extern "C" fn longcat_get_num_frames() -> i32 {
    LONGCAT_NUM_FRAMES.load(std::sync::atomic::Ordering::Relaxed) as i32
}

/// Set video dimensions
#[unsafe(no_mangle)]
pub extern "C" fn longcat_set_video_size(width: i32, height: i32) {
    LONGCAT_VIDEO_WIDTH.store(width.max(64) as usize, std::sync::atomic::Ordering::Relaxed);
    LONGCAT_VIDEO_HEIGHT.store(height.max(64) as usize, std::sync::atomic::Ordering::Relaxed);
}

/// Get video width
#[unsafe(no_mangle)]
pub extern "C" fn longcat_get_video_width() -> i32 {
    LONGCAT_VIDEO_WIDTH.load(std::sync::atomic::Ordering::Relaxed) as i32
}

/// Get video height
#[unsafe(no_mangle)]
pub extern "C" fn longcat_get_video_height() -> i32 {
    LONGCAT_VIDEO_HEIGHT.load(std::sync::atomic::Ordering::Relaxed) as i32
}

/// Set video FPS
#[unsafe(no_mangle)]
pub extern "C" fn longcat_set_video_fps(fps: i32) {
    LONGCAT_VIDEO_FPS.store(fps.max(1) as usize, std::sync::atomic::Ordering::Relaxed);
}

/// Get video FPS
#[unsafe(no_mangle)]
pub extern "C" fn longcat_get_video_fps() -> i32 {
    LONGCAT_VIDEO_FPS.load(std::sync::atomic::Ordering::Relaxed) as i32
}

/// Get the last LongCat error message
#[unsafe(no_mangle)]
pub extern "C" fn longcat_get_error() -> *mut c_char {
    std::ptr::null_mut()
}

/// Free a string returned by LongCat
#[unsafe(no_mangle)]
pub extern "C" fn longcat_free_string(s: *mut c_char) {
    z_image_free_string(s)
}

// ============================================================================
// LongCat Memory Optimization API
// ============================================================================

static LONGCAT_ATTENTION_SLICE_SIZE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Set attention slice size for memory optimization.
#[unsafe(no_mangle)]
pub extern "C" fn longcat_set_attention_slice_size(slice_size: i32) {
    let size = slice_size.max(0) as usize;
    LONGCAT_ATTENTION_SLICE_SIZE.store(size, std::sync::atomic::Ordering::Relaxed);
    set_attention_slice_size(size);
    if size > 0 {
        eprintln!("[longcat] Attention slice size set to {} (memory optimization ON)", size);
    } else {
        eprintln!("[longcat] Attention slicing disabled (full attention)");
    }
}

/// Get the current attention slice size
#[unsafe(no_mangle)]
pub extern "C" fn longcat_get_attention_slice_size() -> i32 {
    LONGCAT_ATTENTION_SLICE_SIZE.load(std::sync::atomic::Ordering::Relaxed) as i32
}

/// Enable low-memory mode with recommended settings.
#[unsafe(no_mangle)]
pub extern "C" fn longcat_enable_low_memory_mode() {
    let config = MemoryConfig::low_memory();
    config.apply();
    LONGCAT_ATTENTION_SLICE_SIZE.store(config.attention_slice_size, std::sync::atomic::Ordering::Relaxed);
    LONGCAT_GUIDANCE_SCALE.store(100, std::sync::atomic::Ordering::Relaxed); // 1.0

    eprintln!("[longcat] LOW MEMORY MODE ENABLED:");
    eprintln!("  - Attention slice size: {}", config.attention_slice_size);
    eprintln!("  - Guidance scale: 1.0 (no CFG, 50% less memory)");
    eprintln!("  - VAE tiling: {}", config.vae_tiling);
}

/// Get estimated memory usage for current settings (in MB)
#[unsafe(no_mangle)]
pub extern "C" fn longcat_estimate_memory_mb() -> i32 {
    let num_frames = LONGCAT_NUM_FRAMES.load(std::sync::atomic::Ordering::Relaxed);
    let height = LONGCAT_VIDEO_HEIGHT.load(std::sync::atomic::Ordering::Relaxed);
    let width = LONGCAT_VIDEO_WIDTH.load(std::sync::atomic::Ordering::Relaxed);
    let slice_size = LONGCAT_ATTENTION_SLICE_SIZE.load(std::sync::atomic::Ordering::Relaxed);

    let latent_t = num_frames / 4;
    let latent_h = height / 8;
    let latent_w = width / 8;
    let num_tokens = latent_t * (latent_h / 2) * (latent_w / 2);
    let model_mb = 27_000;
    let heads = 32;
    let attention_mb = if slice_size > 0 && num_tokens > slice_size {
        (heads * slice_size * num_tokens * 2) / (1024 * 1024)
    } else {
        (heads * num_tokens * num_tokens * 2) / (1024 * 1024)
    };
    let total_mb = model_mb + attention_mb as i32 + 1000;

    eprintln!("[longcat] Memory estimate for {}x{} @ {} frames:", width, height, num_frames);
    eprintln!("  - Sequence length: {} tokens", num_tokens);
    eprintln!("  - Attention per layer: {} MB", attention_mb);
    eprintln!("  - Total estimated: {} MB ({:.1} GB)", total_mb, total_mb as f32 / 1000.0);

    total_mb
}

// ============================================================================
// LongCat Pause/Resume/Cancel/Progress FFI Functions
// ============================================================================

/// Pause the current video generation
#[unsafe(no_mangle)]
pub extern "C" fn longcat_pause() {
    if let Some(control) = LONGCAT_CONTROL.get() {
        control.pause();
        eprintln!("[longcat] Pause requested");
    } else {
        eprintln!("[longcat] No active generation to pause");
    }
}

/// Resume a paused video generation
#[unsafe(no_mangle)]
pub extern "C" fn longcat_resume() {
    if let Some(control) = LONGCAT_CONTROL.get() {
        control.resume();
        eprintln!("[longcat] Resume requested");
    } else {
        eprintln!("[longcat] No active generation to resume");
    }
}

/// Cancel the current video generation
#[unsafe(no_mangle)]
pub extern "C" fn longcat_cancel() {
    if let Some(control) = LONGCAT_CONTROL.get() {
        control.cancel();
        control.resume();
        eprintln!("[longcat] Cancel requested");
    } else {
        eprintln!("[longcat] No active generation to cancel");
    }
}

/// Check if generation is currently paused
#[unsafe(no_mangle)]
pub extern "C" fn longcat_is_paused() -> i32 {
    if let Some(control) = LONGCAT_CONTROL.get() {
        if control.is_paused() { 1 } else { 0 }
    } else {
        0
    }
}

/// Check if generation was cancelled
#[unsafe(no_mangle)]
pub extern "C" fn longcat_is_cancelled() -> i32 {
    if let Some(control) = LONGCAT_CONTROL.get() {
        if control.is_cancelled() { 1 } else { 0 }
    } else {
        0
    }
}

/// Get the current generation progress as a percentage (0.0 - 100.0)
#[unsafe(no_mangle)]
pub extern "C" fn longcat_get_progress() -> f32 {
    if let Some(control) = LONGCAT_CONTROL.get() {
        control.get_progress() * 100.0
    } else {
        0.0
    }
}

/// Get the current step number (0-indexed)
#[unsafe(no_mangle)]
pub extern "C" fn longcat_get_current_step() -> i32 {
    if let Some(control) = LONGCAT_CONTROL.get() {
        control.get_current_step() as i32
    } else {
        -1
    }
}

/// Get the total number of steps
#[unsafe(no_mangle)]
pub extern "C" fn longcat_get_total_steps() -> i32 {
    if let Some(control) = LONGCAT_CONTROL.get() {
        control.get_total_steps() as i32
    } else {
        0
    }
}

/// Reset the generation control for a new generation
#[unsafe(no_mangle)]
pub extern "C" fn longcat_reset_control() {
    let control = LONGCAT_CONTROL.get_or_init(GenerationControl::new);
    control.reset();
    eprintln!("[longcat] Control reset for new generation");
}
