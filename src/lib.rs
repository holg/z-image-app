//! Z-Image FFI Library
//!
//! Provides C-compatible interface for image generation and text chat.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use burn::backend::candle::{Candle, CandleDevice};
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::module::Module;
use half::bf16;
use qwen3_burn::{Qwen3Config, Qwen3ForCausalLM, Qwen3Model, Qwen3Tokenizer};
use z_image::modules::ae::AutoEncoder;
use z_image::modules::transformer::ZImageModel;
use z_image::GenerateFromTextOpts;

type Backend = Candle<bf16, i64>;
type CpuBackend = NdArray<f32>;

// Global device
static DEVICE: OnceLock<CandleDevice> = OnceLock::new();

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

/// Initialize the Metal device. Call once at app startup.
#[unsafe(no_mangle)]
pub extern "C" fn z_image_init() -> i32 {
    eprintln!("[z-image] Initializing Metal device...");
    let device = CandleDevice::metal(0);
    match DEVICE.set(device) {
        Ok(_) => {
            eprintln!("[z-image] Metal device initialized");
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

fn load_image_models_internal(model_dir: &PathBuf, device: &CandleDevice) -> Result<(), String> {
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
    device: &CandleDevice,
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

fn init_text_model_internal(model_dir: &PathBuf, device: &CandleDevice) -> Result<(), String> {
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
// LongCat Video Generation API
// ============================================================================
//
// Three modes of operation:
//
// 1. TEXT-TO-VIDEO (T2V) - Main LongCat purpose
//    - longcat_init() -> longcat_load_models() -> longcat_generate_video()
//    - Only loads LongCat models (DiT + VAE + UMT5)
//    - Most memory efficient for pure video generation
//
// 2. IMAGE-TO-VIDEO (I2V) - Animate existing image
//    - longcat_init() -> longcat_load_models() -> longcat_generate_video_from_image()
//    - User provides image path + text description
//    - Only loads LongCat models
//
// 3. TEXT-TO-IMAGE-TO-VIDEO (T2I2V) - Full pipeline
//    - z_image_init() -> z_image_load_models() + longcat_load_models()
//    - -> longcat_generate_video_from_text_with_zimage()
//    - First generates image with z-image, then animates it
//    - Requires both z-image AND longcat models loaded
//
// ============================================================================

use once_cell::sync::OnceCell as OnceCell2;
use parking_lot::Mutex as ParkingMutex;
use longcat_burn::{GenerateConfig, LongCatPipeline, MemoryConfig, PipelineBuilder, set_attention_slice_size};
use longcat_burn::{WanVae, WanVaeConfig, GenerationControl};

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
///
/// This is the main LongCat use case - pure text-to-video generation.
/// Does NOT require z-image models, only LongCat models.
///
/// Call sequence:
///   longcat_init() -> longcat_load_models() -> longcat_generate_video()
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
            // TODO: Save video frames or encode to MP4
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
            // Check if it was a user cancellation
            if msg.contains("cancelled") {
                eprintln!("[longcat] Generation was cancelled");
                return -3; // Special code for cancelled
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
/// User provides an image file and a text prompt describing the motion.
/// Does NOT require z-image models, only LongCat models.
///
/// NOTE: Currently disabled on Metal backend due to missing 3D convolution support.
/// The VAE encoder requires 3D convolutions which Candle Metal doesn't implement.
///
/// Call sequence:
///   longcat_init() -> longcat_load_models() -> longcat_generate_video_from_image()
#[unsafe(no_mangle)]
pub extern "C" fn longcat_generate_video_from_image(
    _image_path: *const c_char,
    _prompt: *const c_char,
    _output_path: *const c_char,
) -> i32 {
    eprintln!("[longcat] Error: I2V mode is not available on Metal backend");
    eprintln!("[longcat] Reason: VAE encoder requires 3D convolutions which Candle Metal doesn't support");
    eprintln!("[longcat] Workaround: Use T2V mode instead (Text-to-Video)");
    return -2; // Special error code for "not supported"
}

/// Generate video from an existing image (I2V mode) - INTERNAL IMPLEMENTATION
/// Currently disabled due to BF16/F32 mismatch between GPU VAE and CPU backend
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

            // Load weights
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

        // Create image tensor on CPU [1, 3, H, W]
        let image_cpu: burn::tensor::Tensor<CpuBackend, 4> = {
            let tensor_1d: burn::tensor::Tensor<CpuBackend, 1> =
                burn::tensor::Tensor::from_floats(chw_data.as_slice(), &cpu_device);
            tensor_1d.reshape([1, 3, img_height as usize, img_width as usize])
        };

        // Expand to video format [1, 3, 1, H, W]
        let image_video = image_cpu.reshape([1, 3, 1, img_height as usize, img_width as usize]);

        // Encode on CPU
        let latent = cpu_vae.encode_deterministic(image_video);
        eprintln!("[longcat] Encoded latent shape: {:?}", latent.dims());

        // Extract data to transfer to GPU
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

    // Catch panics to prevent app crash
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // Create GPU latent tensor
        let latent_gpu: burn::tensor::Tensor<Backend, 5> = {
            // Compute latent dimensions (VAE compresses 4x8x8)
            let latent_c = 16; // latent channels
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
            // TODO: Save video frames or encode to MP4
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
///
/// This is the full pipeline: text -> z-image -> image -> LongCat -> video.
/// REQUIRES both z-image AND LongCat models to be loaded.
///
/// Call sequence:
///   z_image_init() -> z_image_load_models() + longcat_load_models()
///   -> longcat_generate_video_from_text_with_zimage()
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

    // Get video dimensions for the initial image
    let width = LONGCAT_VIDEO_WIDTH.load(std::sync::atomic::Ordering::Relaxed);
    let height = LONGCAT_VIDEO_HEIGHT.load(std::sync::atomic::Ordering::Relaxed);

    // Step 1: Generate image with z-image
    eprintln!("[longcat] Step 1: Generating initial frame with z-image ({}x{})...", width, height);

    // Create temp path for intermediate image
    let temp_image_path = output_path.with_extension("_temp_frame.png");

    let device = DEVICE.get().expect("Device not initialized");

    // Get z-image models
    let models_mutex = IMAGE_MODELS.get().expect("Image models not initialized");
    let models_guard = models_mutex.lock().expect("Lock error");
    let zimage_models = models_guard.as_ref().expect("z-image models not loaded");

    // Generate the initial frame
    let gen_start = Instant::now();
    let opts = GenerateFromTextOpts {
        prompt: prompt.clone(),
        out_path: temp_image_path.clone(),
        width,
        height,
        num_inference_steps: Some(8), // Fast generation for initial frame
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

    // Release z-image lock
    drop(models_guard);

    // Step 2: Load the generated image and create video
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

    // Convert image to tensor
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

    // Catch panics to prevent app crash
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pipeline.generate_from_image(image_tensor, &prompt, &config)
    }));

    // Clean up temp file
    let _ = std::fs::remove_file(&temp_image_path);

    match result {
        Ok(video) => {
            eprintln!("[longcat] Video generated in {:.2}s", video_start.elapsed().as_secs_f32());
            eprintln!("[longcat] Video shape: {:?}", video.dims());
            eprintln!("[longcat] Output: {:?}", output_path);
            // TODO: Save video frames or encode to MP4
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
///
/// Sliced attention computes attention in chunks, dramatically reducing memory usage.
/// Recommended values:
/// - 0: Full attention (fastest, most memory - NOT recommended for video)
/// - 2048: For 32GB+ VRAM
/// - 1024: For 16-24GB VRAM
/// - 512: For 8-16GB VRAM (recommended for most Macs)
/// - 256: For <8GB VRAM (slow but works)
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
///
/// This enables:
/// - Attention slicing with size 512
/// - Guidance scale set to 1.0 (no CFG, halves memory)
/// - Reduced inference steps
#[unsafe(no_mangle)]
pub extern "C" fn longcat_enable_low_memory_mode() {
    // Apply low memory config
    let config = MemoryConfig::low_memory();
    config.apply();

    // Store the slice size locally
    LONGCAT_ATTENTION_SLICE_SIZE.store(config.attention_slice_size, std::sync::atomic::Ordering::Relaxed);

    // Disable CFG to save memory (runs model only once instead of twice per step)
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

    // VAE compression: 4x8x8
    let latent_t = num_frames / 4;
    let latent_h = height / 8;
    let latent_w = width / 8;

    // Patch size 2
    let num_tokens = latent_t * (latent_h / 2) * (latent_w / 2);

    // Model memory (DiT ~27GB)
    let model_mb = 27_000;

    // Attention memory per layer
    let heads = 32;
    let attention_mb = if slice_size > 0 && num_tokens > slice_size {
        // Sliced: [batch, heads, slice_size, seq_len] * 2 bytes
        (heads * slice_size * num_tokens * 2) / (1024 * 1024)
    } else {
        // Full: [batch, heads, seq, seq] * 2 bytes
        (heads * num_tokens * num_tokens * 2) / (1024 * 1024)
    };

    // Peak estimate (model + one layer activation + latents)
    let total_mb = model_mb + attention_mb as i32 + 1000; // +1GB buffer

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
///
/// The generation will pause after completing the current step.
/// Use longcat_resume() to continue.
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
///
/// The generation will be cancelled at the end of the current step.
/// Note: This will cause the generation function to return with an error.
#[unsafe(no_mangle)]
pub extern "C" fn longcat_cancel() {
    if let Some(control) = LONGCAT_CONTROL.get() {
        control.cancel();
        // Also resume in case it was paused
        control.resume();
        eprintln!("[longcat] Cancel requested");
    } else {
        eprintln!("[longcat] No active generation to cancel");
    }
}

/// Check if generation is currently paused
///
/// Returns: 1 if paused, 0 if running or not started
#[unsafe(no_mangle)]
pub extern "C" fn longcat_is_paused() -> i32 {
    if let Some(control) = LONGCAT_CONTROL.get() {
        if control.is_paused() { 1 } else { 0 }
    } else {
        0
    }
}

/// Check if generation was cancelled
///
/// Returns: 1 if cancelled, 0 otherwise
#[unsafe(no_mangle)]
pub extern "C" fn longcat_is_cancelled() -> i32 {
    if let Some(control) = LONGCAT_CONTROL.get() {
        if control.is_cancelled() { 1 } else { 0 }
    } else {
        0
    }
}

/// Get the current generation progress as a percentage (0.0 - 100.0)
///
/// Returns: Progress percentage, or 0.0 if no generation active
#[unsafe(no_mangle)]
pub extern "C" fn longcat_get_progress() -> f32 {
    if let Some(control) = LONGCAT_CONTROL.get() {
        control.get_progress() * 100.0
    } else {
        0.0
    }
}

/// Get the current step number (0-indexed)
///
/// Returns: Current step, or -1 if no generation active
#[unsafe(no_mangle)]
pub extern "C" fn longcat_get_current_step() -> i32 {
    if let Some(control) = LONGCAT_CONTROL.get() {
        control.get_current_step() as i32
    } else {
        -1
    }
}

/// Get the total number of steps
///
/// Returns: Total steps, or 0 if no generation active
#[unsafe(no_mangle)]
pub extern "C" fn longcat_get_total_steps() -> i32 {
    if let Some(control) = LONGCAT_CONTROL.get() {
        control.get_total_steps() as i32
    } else {
        0
    }
}

/// Reset the generation control for a new generation
///
/// Call this before starting a new generation to clear any previous cancel state.
#[unsafe(no_mangle)]
pub extern "C" fn longcat_reset_control() {
    let control = LONGCAT_CONTROL.get_or_init(GenerationControl::new);
    control.reset();
    eprintln!("[longcat] Control reset for new generation");
}
