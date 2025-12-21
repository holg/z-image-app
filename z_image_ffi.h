#ifndef Z_IMAGE_FFI_H
#define Z_IMAGE_FFI_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Generation settings structure
typedef struct {
    int32_t num_inference_steps;  // 4-20, default 8
    int64_t seed;                 // 0 for random
    bool use_attention_slicing;   // Memory optimization
    int32_t slice_size;           // Attention slice size (0 for auto)
    bool low_memory_mode;         // Unload text encoder during diffusion
} ZImageGenerationSettings;

// Create default generation settings
ZImageGenerationSettings z_image_default_settings(void);

// Load image generation models into memory
// model_dir: path to directory containing .bpk model files
// Returns: 0 on success, negative error code on failure
int32_t z_image_load_models(const char* model_dir);

// Unload image generation models from memory
// Returns: 0 on success, negative error code on failure
int32_t z_image_unload_models(void);

// Check if image models are loaded
// Returns: true if models are loaded
bool z_image_models_loaded(void);

// Generate an image from a text prompt
// prompt: text description of the image to generate
// width: image width (multiple of 16, e.g., 512, 768, 1024)
// height: image height (multiple of 16)
// model_dir: path to directory containing model files
// output_path: path where to save the generated PNG image
// settings: generation settings (can be NULL for defaults)
// Returns: 0 on success, negative error code on failure
int32_t z_image_generate(
    const char* prompt,
    int32_t width,
    int32_t height,
    const char* model_dir,
    const char* output_path,
    const ZImageGenerationSettings* settings
);

// Generate image with cached models (must call z_image_load_models first)
// prompt: text description
// width: image width
// height: image height
// output_path: output PNG path
// settings: generation settings (can be NULL for defaults)
// Returns: 0 on success, negative error code on failure
int32_t z_image_generate_cached(
    const char* prompt,
    int32_t width,
    int32_t height,
    const char* output_path,
    const ZImageGenerationSettings* settings
);

// Load chat model (Qwen3-0.6B)
// model_dir: path to directory containing chat model files
// Returns: 0 on success, negative error code on failure
int32_t z_image_load_chat_model(const char* model_dir);

// Unload chat model from memory
// Returns: 0 on success, negative error code on failure
int32_t z_image_unload_chat_model(void);

// Check if chat model is loaded
// Returns: true if chat model is loaded
bool z_image_chat_model_loaded(void);

// Generate chat response
// prompt: user message
// max_tokens: maximum tokens to generate
// output_buffer: buffer to store the response
// buffer_size: size of output buffer
// Returns: length of response on success, negative error code on failure
int32_t z_image_chat(
    const char* prompt,
    int32_t max_tokens,
    char* output_buffer,
    int32_t buffer_size
);

// Get last error message
// Returns: pointer to error string (valid until next call)
const char* z_image_get_last_error(void);

// Get library version
// Returns: version string
const char* z_image_version(void);

#ifdef __cplusplus
}
#endif

#endif // Z_IMAGE_FFI_H
