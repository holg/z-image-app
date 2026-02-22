//! Z-Image FFI Library
//!
//! Provides C-compatible interface for image generation and text chat.

#![recursion_limit = "256"]

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use qwen3_burn::{Qwen3Config, Qwen3ForCausalLM, Qwen3Model, Qwen3Tokenizer};
use z_image::modules::ae::AutoEncoder;
use z_image::modules::transformer::ZImageModel;
use z_image::GenerateFromTextOpts;

// Backend selection based on compile-time features
#[cfg(feature = "metal")]
mod backend {
    use burn::backend::candle::{Candle, CandleDevice};
    use half::bf16;
    pub type Backend = Candle<bf16, i64>;
    pub type Device = CandleDevice;
    pub fn create_device() -> Device { CandleDevice::metal(0) }
    pub const BACKEND_NAME: &str = "Metal (Candle)";
}

#[cfg(feature = "cuda")]
#[cfg(not(feature = "metal"))]
mod backend {
    use burn::backend::candle::{Candle, CandleDevice};
    use half::bf16;
    pub type Backend = Candle<bf16, i64>;
    pub type Device = CandleDevice;
    pub fn create_device() -> Device { CandleDevice::cuda(0) }
    pub const BACKEND_NAME: &str = "CUDA (Candle)";
}

#[cfg(any(feature = "wgpu", feature = "wgpu-metal", feature = "vulkan"))]
#[cfg(not(any(feature = "metal", feature = "cuda")))]
mod backend {
    use burn::backend::wgpu::{Wgpu, WgpuDevice};
    pub type Backend = Wgpu<f32, i32>;
    pub type Device = WgpuDevice;
    pub fn create_device() -> Device { WgpuDevice::default() }
    pub const BACKEND_NAME: &str = "WGPU";
}

#[cfg(feature = "cpu")]
#[cfg(not(any(feature = "metal", feature = "cuda", feature = "wgpu", feature = "wgpu-metal", feature = "vulkan")))]
mod backend {
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    pub type Backend = NdArray<f32>;
    pub type Device = NdArrayDevice;
    pub fn create_device() -> Device { NdArrayDevice::Cpu }
    pub const BACKEND_NAME: &str = "CPU (NdArray)";
}

#[cfg(not(any(feature = "metal", feature = "cuda", feature = "wgpu", feature = "wgpu-metal", feature = "vulkan", feature = "cpu")))]
mod backend {
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    pub type Backend = NdArray<f32>;
    pub type Device = NdArrayDevice;
    pub fn create_device() -> Device { NdArrayDevice::Cpu }
    pub const BACKEND_NAME: &str = "CPU (fallback)";
}

type Backend = backend::Backend;

#[cfg(feature = "video")]
type CpuBackend = burn::backend::ndarray::NdArray<f32>;

// Global device
static DEVICE: OnceLock<backend::Device> = OnceLock::new();

// Cached image models
struct ImageModels {
    tokenizer: Qwen3Tokenizer,
    text_encoder: Qwen3Model<Backend>,
    transformer: ZImageModel<Backend>,
    autoencoder: AutoEncoder<Backend>,
}

static IMAGE_MODELS: OnceLock<Mutex<Option<ImageModels>>> = OnceLock::new();

// Cached text model
static TEXT_MODEL: OnceLock<Mutex<Option<Qwen3ForCausalLM<Backend>>>> = OnceLock::new();
static TEXT_TOKENIZER: OnceLock<Mutex<Option<Qwen3Tokenizer>>> = OnceLock::new();

/// Initialize the compute device. Call once at app startup.
#[unsafe(no_mangle)]
pub extern "C" fn z_image_init() -> i32 {
    eprintln!("[z-image] Initializing {} device...", backend::BACKEND_NAME);
    let device = backend::create_device();
    match DEVICE.set(device) {
        Ok(_) => {
            eprintln!("[z-image] {} device initialized", backend::BACKEND_NAME);
            0
        }
        Err(_) => {
            eprintln!("[z-image] Device already initialized");
            0
        }
    }
}

/// Pre-load image generation models into GPU memory.
#[unsafe(no_mangle)]
pub extern "C" fn z_image_load_models(model_dir: *const c_char) -> i32 {
    let model_dir = unsafe {
        if model_dir.is_null() {
            eprintln!("[z-image] Error: model_dir is null");
            return -1;
        }
        match CStr::from_ptr(model_dir).to_str() {
            Ok(s) => PathBuf::from(s),
            Err(_) => {
                eprintln!("[z-image] Error: Invalid UTF-8 in model_dir");
                return -1;
            }
        }
    };

    let device = match DEVICE.get() {
        Some(d) => d,
        None => {
            eprintln!("[z-image] Error: Device not initialized. Call z_image_init first.");
            return -1;
        }
    };

    match load_image_models_internal(&model_dir, device) {
        Ok(_) => 0,
        Err(e) => {
            eprintln!("[z-image] Error loading models: {}", e);
            -1
        }
    }
}

fn load_image_models_internal(model_dir: &PathBuf, device: &backend::Device) -> Result<(), String> {
    let total_start = Instant::now();
    eprintln!("[z-image] Loading models from {:?}...", model_dir);

    // Check if model directory exists
    if !model_dir.exists() {
        let err = format!("Model directory not found: {:?}", model_dir);
        set_last_error(&err);
        return Err(err);
    }

    // Build paths - prefer .bpk files, fallback to .safetensors
    let transformer_path = model_dir.join("z_image_turbo_bf16.bpk");
    let tokenizer_path = model_dir.join("qwen3_tokenizer.json");
    let tokenizer_path_alt = model_dir.join("qwen3-tokenizer.json"); // Alternative name

    // Find tokenizer
    let tokenizer_path = if tokenizer_path.exists() {
        tokenizer_path
    } else if tokenizer_path_alt.exists() {
        tokenizer_path_alt
    } else {
        let err = format!("Tokenizer not found. Please download 'qwen3_tokenizer.json' to {:?}", model_dir);
        set_last_error(&err);
        return Err(err);
    };

    // Check transformer
    if !transformer_path.exists() {
        let err = format!("ZImage model not found: {:?}. Download from holgt/z-image-turbo-burn", transformer_path);
        set_last_error(&err);
        return Err(err);
    }

    // Text encoder: prefer .bpk, fallback to .safetensors
    let te_bpk = model_dir.join("qwen3_4b_text_encoder.bpk");
    let te_safetensors = model_dir.join("qwen3_4b_text_encoder.safetensors");
    let text_encoder_path = if te_bpk.exists() {
        te_bpk
    } else if te_safetensors.exists() {
        te_safetensors
    } else {
        let err = format!("Text encoder not found. Please download 'qwen3_4b_text_encoder.bpk' from holgt/qwen3-4b-text-encoder-burn to {:?}", model_dir);
        set_last_error(&err);
        return Err(err);
    };

    // Load tokenizer
    let t0 = Instant::now();
    eprintln!("[z-image] Loading tokenizer from {:?}...", tokenizer_path);
    let tokenizer = Qwen3Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| {
            let err = format!("Failed to load tokenizer from {:?}: {}", tokenizer_path, e);
            set_last_error(&err);
            err
        })?;
    eprintln!("[z-image] Tokenizer loaded in {:.2}s", t0.elapsed().as_secs_f32());

    // Load text encoder
    let t0 = Instant::now();
    eprintln!("[z-image] Loading text encoder from {:?}...", text_encoder_path);
    let mut text_encoder: Qwen3Model<Backend> = Qwen3Config::z_image_text_encoder().init(device);
    text_encoder.load_weights(&text_encoder_path)
        .map_err(|e| {
            let err = format!("Failed to load text encoder from {:?}: {:?}", text_encoder_path, e);
            set_last_error(&err);
            err
        })?;
    eprintln!("[z-image] Text encoder loaded in {:.2}s", t0.elapsed().as_secs_f32());

    // Load transformer
    let t0 = Instant::now();
    eprintln!("[z-image] Loading transformer from {:?}...", transformer_path);
    use z_image::modules::transformer::ZImageModelConfig;
    let mut transformer: ZImageModel<Backend> = ZImageModelConfig::default().init(device);
    transformer.load_weights(&transformer_path)
        .map_err(|e| {
            let err = format!("Failed to load transformer from {:?}: {:?}", transformer_path, e);
            set_last_error(&err);
            err
        })?;
    eprintln!("[z-image] Transformer loaded in {:.2}s", t0.elapsed().as_secs_f32());

    // Load autoencoder
    let t0 = Instant::now();
    let ae_bpk = model_dir.join("ae.bpk");
    let ae_safetensors = model_dir.join("ae.safetensors");
    let ae_path = if ae_bpk.exists() { ae_bpk } else { ae_safetensors };
    eprintln!("[z-image] Loading autoencoder from {:?}...", ae_path);
    use z_image::modules::ae::AutoEncoderConfig;
    let mut autoencoder: AutoEncoder<Backend> = AutoEncoderConfig::flux_ae().init(device);
    autoencoder.load_weights(&ae_path)
        .map_err(|e| {
            let err = format!("Failed to load autoencoder from {:?}: {:?}", ae_path, e);
            set_last_error(&err);
            err
        })?;
    eprintln!("[z-image] Autoencoder loaded in {:.2}s", t0.elapsed().as_secs_f32());

    let total_time = total_start.elapsed().as_secs_f32();
    eprintln!("[z-image] All models loaded in {:.2}s", total_time);

    // Store in global cache
    let models = ImageModels {
        tokenizer,
        text_encoder,
        transformer,
        autoencoder,
    };

    let models_mutex = IMAGE_MODELS.get_or_init(|| Mutex::new(None));
    *models_mutex.lock().unwrap() = Some(models);

    Ok(())
}

/// Check if image generation models are loaded.
#[unsafe(no_mangle)]
pub extern "C" fn z_image_models_loaded() -> i32 {
    if let Some(models_mutex) = IMAGE_MODELS.get() {
        if let Ok(guard) = models_mutex.lock() {
            if guard.is_some() {
                return 1;
            }
        }
    }
    0
}

/// Unload image generation models from GPU memory.
#[unsafe(no_mangle)]
pub extern "C" fn z_image_unload_models() -> i32 {
    if let Some(models_mutex) = IMAGE_MODELS.get() {
        if let Ok(mut guard) = models_mutex.lock() {
            if guard.is_some() {
                eprintln!("[z-image] Unloading image models...");
                *guard = None;
                eprintln!("[z-image] Image models unloaded - GPU memory freed");
                return 0;
            }
        }
    }
    eprintln!("[z-image] No models to unload");
    0
}

/// Generate an image from a text prompt.
#[unsafe(no_mangle)]
pub extern "C" fn z_image_generate(
    prompt: *const c_char,
    output_path: *const c_char,
    model_dir: *const c_char,
    width: i32,
    height: i32,
) -> i32 {
    let prompt = unsafe {
        if prompt.is_null() {
            return -1;
        }
        match CStr::from_ptr(prompt).to_str() {
            Ok(s) => s.to_string(),
            Err(_) => return -1,
        }
    };

    let output_path = unsafe {
        if output_path.is_null() {
            return -1;
        }
        match CStr::from_ptr(output_path).to_str() {
            Ok(s) => PathBuf::from(s),
            Err(_) => return -1,
        }
    };

    let model_dir = unsafe {
        if model_dir.is_null() {
            return -1;
        }
        match CStr::from_ptr(model_dir).to_str() {
            Ok(s) => PathBuf::from(s),
            Err(_) => return -1,
        }
    };

    let device = match DEVICE.get() {
        Some(d) => d,
        None => {
            eprintln!("[z-image] Error: Device not initialized");
            return -1;
        }
    };

    match generate_image_internal(
        &prompt,
        &output_path,
        &model_dir,
        width as usize,
        height as usize,
        device,
    ) {
        Ok(_) => 0,
        Err(e) => {
            eprintln!("[z-image] Generation error: {}", e);
            -1
        }
    }
}

fn generate_image_internal(
    prompt: &str,
    output_path: &PathBuf,
    model_dir: &PathBuf,
    width: usize,
    height: usize,
    device: &backend::Device,
) -> Result<(), String> {
    let total_start = Instant::now();

    eprintln!("[z-image] Generating image for prompt: {}", prompt);
    eprintln!("[z-image] Output: {:?}", output_path);
    eprintln!("[z-image] Model dir: {:?}", model_dir);
    eprintln!("[z-image] Size: {}x{}", width, height);

    // Build paths - prefer .bpk files, fallback to .safetensors
    let transformer_path = model_dir.join("z_image_turbo_bf16.bpk");
    let tokenizer_path = model_dir.join("qwen3-tokenizer.json");

    // Autoencoder: prefer .bpk, fallback to .safetensors
    let ae_bpk = model_dir.join("ae.bpk");
    let ae_safetensors = model_dir.join("ae.safetensors");
    let ae_path = if ae_bpk.exists() { ae_bpk } else { ae_safetensors };

    // Text encoder: prefer .bpk, fallback to .safetensors
    let te_bpk = model_dir.join("qwen3_4b_text_encoder.bpk");
    let te_safetensors = model_dir.join("qwen3_4b_text_encoder.safetensors");
    let text_encoder_path = if te_bpk.exists() { te_bpk } else { te_safetensors };

    // Generate image
    // Check if models are cached
    if let Some(models_mutex) = IMAGE_MODELS.get() {
        if let Ok(guard) = models_mutex.lock() {
            if let Some(models) = guard.as_ref() {
                eprintln!("[z-image] Using cached models");

                // Tokenize prompt
                let t0 = Instant::now();
                let formatted_prompt = models.tokenizer.apply_chat_template(prompt);
                let (input_ids, _) = models.tokenizer.encode(&formatted_prompt)
                    .map_err(|e| format!("Tokenization failed: {}", e))?;
                let num_tokens = input_ids.len();
                eprintln!("[z-image] Prompt tokenized: {} tokens in {:.3}s", num_tokens, t0.elapsed().as_secs_f32());
                eprintln!("[z-image] Prompt: \"{}\"", prompt);

                // Get generation settings
                let num_steps = NUM_INFERENCE_STEPS.load(std::sync::atomic::Ordering::Relaxed);
                let seed = if USE_SEED.load(std::sync::atomic::Ordering::Relaxed) {
                    Some(GENERATION_SEED.load(std::sync::atomic::Ordering::Relaxed))
                } else {
                    None
                };

                // Generate
                eprintln!("[z-image] Starting generation ({}x{}, {} steps)...", width, height, num_steps);
                let gen_start = Instant::now();

                let opts = GenerateFromTextOpts {
                    prompt: prompt.to_string(),
                    out_path: output_path.clone(),
                    width,
                    height,
                    num_inference_steps: Some(num_steps),
                    seed,
                };
                z_image::generate_from_text(
                    &opts,
                    &models.tokenizer,
                    &models.text_encoder,
                    &models.autoencoder,
                    &models.transformer,
                    device,
                ).map_err(|e| format!("Generation failed: {:?}", e))?;

                let gen_time = gen_start.elapsed().as_secs_f32();
                let total_time = total_start.elapsed().as_secs_f32();

                eprintln!("[z-image] ========== STATISTICS ==========");
                eprintln!("[z-image] Prompt tokens: {}", num_tokens);
                eprintln!("[z-image] Image size: {}x{}", width, height);
                eprintln!("[z-image] Using cached models: YES");
                eprintln!("[z-image] Generation time: {:.2}s", gen_time);
                eprintln!("[z-image] Total time: {:.2}s", total_time);
                eprintln!("[z-image] ================================");

                return Ok(());
            }
        }
    }

    // Load models on demand (slower path)
    let low_memory = LOW_MEMORY_MODE.load(std::sync::atomic::Ordering::Relaxed);

    if low_memory {
        eprintln!("[z-image] LOW MEMORY MODE: Loading models sequentially to minimize peak memory");
    } else {
        eprintln!("[z-image] Loading models on demand (consider using z_image_load_models for faster generation)");
    }

    let load_start = Instant::now();

    // Load tokenizer (always needed)
    let tokenizer = Qwen3Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| format!("Failed to load tokenizer: {}", e))?;

    // Tokenize prompt first
    let t0 = Instant::now();
    let formatted_prompt = tokenizer.apply_chat_template(prompt);
    let (input_ids, _) = tokenizer.encode(&formatted_prompt)
        .map_err(|e| format!("Tokenization failed: {}", e))?;
    let num_tokens = input_ids.len();
    eprintln!("[z-image] Prompt tokenized: {} tokens in {:.3}s", num_tokens, t0.elapsed().as_secs_f32());
    eprintln!("[z-image] Prompt: \"{}\"", prompt);

    // Compute prompt embedding using text encoder
    let t0 = Instant::now();
    eprintln!("[z-image] Loading text encoder...");
    let mut text_encoder: Qwen3Model<Backend> = Qwen3Config::z_image_text_encoder().init(device);
    text_encoder.load_weights(&text_encoder_path)
        .map_err(|e| format!("Failed to load text encoder: {:?}", e))?;
    eprintln!("[z-image] Text encoder loaded in {:.2}s", t0.elapsed().as_secs_f32());

    let t0 = Instant::now();
    eprintln!("[z-image] Computing prompt embedding...");
    let prompt_embedding = z_image::compute_prompt_embedding(prompt, &tokenizer, &text_encoder, device)
        .map_err(|e| format!("Failed to compute embedding: {:?}", e))?;
    eprintln!("[z-image] Prompt embedding computed in {:.2}s", t0.elapsed().as_secs_f32());

    // In low memory mode, drop text encoder before loading other models
    if low_memory {
        eprintln!("[z-image] LOW MEMORY: Unloading text encoder to free ~7.5GB...");
        drop(text_encoder);
    }

    // Load transformer
    let t0 = Instant::now();
    eprintln!("[z-image] Loading transformer...");
    use z_image::modules::transformer::ZImageModelConfig;
    let mut transformer: ZImageModel<Backend> = ZImageModelConfig::default().init(device);
    transformer.load_weights(&transformer_path)
        .map_err(|e| format!("Failed to load transformer: {:?}", e))?;
    eprintln!("[z-image] Transformer loaded in {:.2}s", t0.elapsed().as_secs_f32());

    // Load autoencoder
    let t0 = Instant::now();
    eprintln!("[z-image] Loading autoencoder...");
    use z_image::modules::ae::AutoEncoderConfig;
    let mut ae: AutoEncoder<Backend> = AutoEncoderConfig::flux_ae().init(device);
    ae.load_weights(&ae_path)
        .map_err(|e| format!("Failed to load autoencoder: {:?}", e))?;
    eprintln!("[z-image] Autoencoder loaded in {:.2}s", t0.elapsed().as_secs_f32());

    let load_time = load_start.elapsed().as_secs_f32();
    eprintln!("[z-image] All models processed in {:.2}s", load_time);

    // Get generation settings
    let num_steps = NUM_INFERENCE_STEPS.load(std::sync::atomic::Ordering::Relaxed);
    let seed = if USE_SEED.load(std::sync::atomic::Ordering::Relaxed) {
        Some(GENERATION_SEED.load(std::sync::atomic::Ordering::Relaxed))
    } else {
        None
    };

    // Generate using pre-computed embedding
    eprintln!("[z-image] Starting generation ({}x{}, {} steps)...", width, height, num_steps);
    let gen_start = Instant::now();

    let gen_opts = z_image::GenerateWithEmbeddingOpts {
        width,
        height,
        num_inference_steps: Some(num_steps),
        seed,
    };
    z_image::generate_with_embedding(
        prompt_embedding,
        output_path,
        &gen_opts,
        &ae,
        &transformer,
        device,
    ).map_err(|e| format!("Generation failed: {:?}", e))?;

    let gen_time = gen_start.elapsed().as_secs_f32();
    let total_time = total_start.elapsed().as_secs_f32();

    eprintln!("[z-image] ========== STATISTICS ==========");
    eprintln!("[z-image] Prompt tokens: {}", num_tokens);
    eprintln!("[z-image] Image size: {}x{}", width, height);
    eprintln!("[z-image] Inference steps: {}", num_steps);
    eprintln!("[z-image] Using cached models: NO");
    eprintln!("[z-image] Low memory mode: {}", if low_memory { "YES" } else { "NO" });
    eprintln!("[z-image] Model load time: {:.2}s", load_time);
    eprintln!("[z-image] Generation time: {:.2}s", gen_time);
    eprintln!("[z-image] Total time: {:.2}s", total_time);
    eprintln!("[z-image] ================================");

    Ok(())
}

// Thread-local storage for last error message
static LAST_ERROR: OnceLock<Mutex<Option<String>>> = OnceLock::new();

fn set_last_error(error: &str) {
    let mutex = LAST_ERROR.get_or_init(|| Mutex::new(None));
    *mutex.lock().unwrap() = Some(error.to_string());
}

fn get_last_error() -> Option<String> {
    LAST_ERROR.get().and_then(|m| m.lock().ok()).and_then(|g| g.clone())
}

/// Get the last error message.
#[unsafe(no_mangle)]
pub extern "C" fn z_image_get_error() -> *mut c_char {
    match get_last_error() {
        Some(error) => {
            match CString::new(error) {
                Ok(c_str) => c_str.into_raw(),
                Err(_) => std::ptr::null_mut(),
            }
        }
        None => std::ptr::null_mut(),
    }
}

/// Free a string returned by this library.
#[unsafe(no_mangle)]
pub extern "C" fn z_image_free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe {
            let _ = CString::from_raw(s);
        }
    }
}

// ============================================================================
// Text Generation API (Qwen3-0.6B)
// ============================================================================

/// Initialize the text generation model.
#[unsafe(no_mangle)]
pub extern "C" fn qwen3_init(model_dir: *const c_char) -> i32 {
    let model_dir = unsafe {
        if model_dir.is_null() {
            eprintln!("[qwen3] Error: model_dir is null");
            return -1;
        }
        match CStr::from_ptr(model_dir).to_str() {
            Ok(s) => PathBuf::from(s),
            Err(_) => {
                eprintln!("[qwen3] Error: Invalid UTF-8 in model_dir");
                return -1;
            }
        }
    };

    let device = match DEVICE.get() {
        Some(d) => d,
        None => {
            eprintln!("[qwen3] Error: Device not initialized. Call z_image_init first.");
            return -1;
        }
    };

    match init_text_model_internal(&model_dir, device) {
        Ok(_) => 0,
        Err(e) => {
            eprintln!("[qwen3] Error: {}", e);
            -1
        }
    }
}

fn init_text_model_internal(model_dir: &PathBuf, device: &backend::Device) -> Result<(), String> {
    eprintln!("[qwen3] Loading Qwen3-0.6B from {:?}...", model_dir);

    // Load tokenizer
    let tokenizer_path = model_dir.join("tokenizer.json");
    eprintln!("[qwen3] Loading tokenizer from {:?}...", tokenizer_path);
    let tokenizer = Qwen3Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| format!("Failed to load tokenizer: {}", e))?;

    // Store tokenizer
    let tokenizer_mutex = TEXT_TOKENIZER.get_or_init(|| Mutex::new(None));
    *tokenizer_mutex.lock().unwrap() = Some(tokenizer);
    eprintln!("[qwen3] Tokenizer loaded");

    // Load model - prefer .bpk
    let bpk_path = model_dir.join("model.bpk");
    let safetensors_path = model_dir.join("model.safetensors");
    let model_path = if bpk_path.exists() { bpk_path } else { safetensors_path };

    eprintln!("[qwen3] Loading model from {:?}...", model_path);
    let mut model: Qwen3ForCausalLM<Backend> = Qwen3Config::qwen3_0_6b().init_causal_lm(device);
    model.load_weights(&model_path)
        .map_err(|e| format!("Failed to load model: {:?}", e))?;

    // Store model
    let model_mutex = TEXT_MODEL.get_or_init(|| Mutex::new(None));
    *model_mutex.lock().unwrap() = Some(model);

    eprintln!("[qwen3] Model loaded successfully");
    Ok(())
}

/// Check if text generation model is loaded.
#[unsafe(no_mangle)]
pub extern "C" fn qwen3_is_loaded() -> i32 {
    if let Some(model_mutex) = TEXT_MODEL.get() {
        if let Ok(guard) = model_mutex.lock() {
            if guard.is_some() {
                return 1;
            }
        }
    }
    0
}

/// Generate text from a prompt.
#[unsafe(no_mangle)]
pub extern "C" fn qwen3_generate(
    prompt: *const c_char,
    max_tokens: i32,
    temperature: f32,
) -> *mut c_char {
    let prompt = unsafe {
        if prompt.is_null() {
            return std::ptr::null_mut();
        }
        match CStr::from_ptr(prompt).to_str() {
            Ok(s) => s.to_string(),
            Err(_) => return std::ptr::null_mut(),
        }
    };

    match generate_text_internal(&prompt, max_tokens as usize, temperature) {
        Ok(result) => {
            match CString::new(result) {
                Ok(c_str) => c_str.into_raw(),
                Err(_) => std::ptr::null_mut(),
            }
        }
        Err(e) => {
            eprintln!("[qwen3] Generation error: {}", e);
            std::ptr::null_mut()
        }
    }
}

fn generate_text_internal(prompt: &str, max_tokens: usize, temperature: f32) -> Result<String, String> {
    use burn::tensor::{Int, Tensor};

    let device = DEVICE.get().ok_or("Device not initialized")?;

    // Get tokenizer
    let tokenizer_mutex = TEXT_TOKENIZER.get().ok_or("Tokenizer not loaded")?;
    let tokenizer_guard = tokenizer_mutex.lock().map_err(|_| "Lock error")?;
    let tokenizer = tokenizer_guard.as_ref().ok_or("Tokenizer not loaded")?;

    // Get model
    let model_mutex = TEXT_MODEL.get().ok_or("Model not loaded")?;
    let model_guard = model_mutex.lock().map_err(|_| "Lock error")?;
    let model = model_guard.as_ref().ok_or("Model not loaded")?;

    // Format prompt with chat template
    let formatted = tokenizer.apply_chat_template(prompt);

    // Tokenize
    let (input_ids, _) = tokenizer.encode_no_pad(&formatted)
        .map_err(|e| format!("Tokenization failed: {}", e))?;

    eprintln!("[qwen3] Input: {} tokens", input_ids.len());

    // Create input tensor
    let input_ids_i64: Vec<i64> = input_ids.iter().map(|&x| x as i64).collect();
    let input_tensor: Tensor<Backend, 2, Int> =
        Tensor::<Backend, 1, Int>::from_data(input_ids_i64.as_slice(), device)
            .reshape([1, input_ids.len()]);

    // Generate
    let output = model.generate_with_cache(
        input_tensor,
        max_tokens,
        temperature,
        0.9,
        50,
    );

    // Decode output
    let output_data = output.into_data();
    let output_ids: Vec<u32> = output_data
        .as_slice::<i64>()
        .unwrap_or(&[])
        .iter()
        .skip(input_ids.len())
        .map(|&x| x as u32)
        .collect();

    let text = tokenizer.decode(&output_ids).unwrap_or_default();

    // Clean up special tokens
    let mut clean_text = text;
    if let Some(pos) = clean_text.find("<|im_end|>") {
        clean_text = clean_text[..pos].to_string();
    }
    if let Some(pos) = clean_text.find("<|endoftext|>") {
        clean_text = clean_text[..pos].to_string();
    }

    Ok(clean_text.trim().to_string())
}

/// Unload text generation model from GPU memory.
#[unsafe(no_mangle)]
pub extern "C" fn qwen3_unload() -> i32 {
    if let Some(model_mutex) = TEXT_MODEL.get() {
        if let Ok(mut guard) = model_mutex.lock() {
            if guard.is_some() {
                eprintln!("[qwen3] Unloading text generation model...");
                *guard = None;
                eprintln!("[qwen3] Text model unloaded - GPU memory freed");
                return 0;
            }
        }
    }
    eprintln!("[qwen3] No model to unload");
    0
}

// ============================================================================
// Memory Optimization API
// ============================================================================

/// Set the attention slice size for memory optimization.
#[unsafe(no_mangle)]
pub extern "C" fn z_image_set_attention_slice_size(slice_size: i32) -> i32 {
    z_image::set_attention_slice_size(slice_size as usize);
    0
}

/// Get the current attention slice size.
#[unsafe(no_mangle)]
pub extern "C" fn z_image_get_attention_slice_size() -> i32 {
    z_image::get_attention_slice_size() as i32
}

/// Enable or disable low memory mode for image generation.
#[unsafe(no_mangle)]
pub extern "C" fn z_image_set_low_memory_mode(enabled: i32) -> i32 {
    LOW_MEMORY_MODE.store(enabled != 0, std::sync::atomic::Ordering::Relaxed);
    if enabled != 0 {
        eprintln!("[z-image] Low memory mode ENABLED - text encoder will be unloaded during diffusion");
    } else {
        eprintln!("[z-image] Low memory mode DISABLED - all models stay in memory");
    }
    0
}

/// Check if low memory mode is enabled.
#[unsafe(no_mangle)]
pub extern "C" fn z_image_get_low_memory_mode() -> i32 {
    if LOW_MEMORY_MODE.load(std::sync::atomic::Ordering::Relaxed) { 1 } else { 0 }
}

static LOW_MEMORY_MODE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

// Generation settings
static NUM_INFERENCE_STEPS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(8);
static GENERATION_SEED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static USE_SEED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Set the number of inference steps for image generation.
#[unsafe(no_mangle)]
pub extern "C" fn z_image_set_num_steps(steps: i32) -> i32 {
    let steps = steps.max(1).min(50) as usize;
    NUM_INFERENCE_STEPS.store(steps, std::sync::atomic::Ordering::Relaxed);
    eprintln!("[z-image] Number of inference steps set to {}", steps);
    0
}

/// Get the current number of inference steps.
#[unsafe(no_mangle)]
pub extern "C" fn z_image_get_num_steps() -> i32 {
    NUM_INFERENCE_STEPS.load(std::sync::atomic::Ordering::Relaxed) as i32
}

/// Set the random seed for reproducible generation.
#[unsafe(no_mangle)]
pub extern "C" fn z_image_set_seed(seed: u64) -> i32 {
    if seed == 0 {
        USE_SEED.store(false, std::sync::atomic::Ordering::Relaxed);
        eprintln!("[z-image] Random seed disabled (using random)");
    } else {
        GENERATION_SEED.store(seed, std::sync::atomic::Ordering::Relaxed);
        USE_SEED.store(true, std::sync::atomic::Ordering::Relaxed);
        eprintln!("[z-image] Random seed set to {}", seed);
    }
    0
}

/// Get the current random seed.
#[unsafe(no_mangle)]
pub extern "C" fn z_image_get_seed() -> u64 {
    if USE_SEED.load(std::sync::atomic::Ordering::Relaxed) {
        GENERATION_SEED.load(std::sync::atomic::Ordering::Relaxed)
    } else {
        0
    }
}

// ============================================================================
// LongCat Video Generation API (optional, feature-gated behind "video")
// ============================================================================

#[cfg(feature = "video")]
mod longcat_ffi;

