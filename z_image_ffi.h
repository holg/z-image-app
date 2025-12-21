#ifndef Z_IMAGE_FFI_H
#define Z_IMAGE_FFI_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Initialization
// ============================================================================

// Initialize the GPU device (Metal on macOS, CUDA on Windows/Linux)
// Must be called before any other functions
// Returns: 0 on success, negative error code on failure
int32_t z_image_init(void);

// ============================================================================
// Image Generation Models
// ============================================================================

// Load image generation models into GPU memory
// model_dir: path to directory containing .bpk model files
// Returns: 0 on success, negative error code on failure
int32_t z_image_load_models(const char* model_dir);

// Unload image generation models from GPU memory
// Returns: 0 on success
int32_t z_image_unload_models(void);

// Check if image models are loaded
// Returns: 1 if models are loaded, 0 otherwise
int32_t z_image_models_loaded(void);

// ============================================================================
// Image Generation
// ============================================================================

// Generate an image from a text prompt
// prompt: text description of the image to generate
// output_path: path where to save the generated PNG image
// model_dir: path to directory containing model files
// width: image width (multiple of 16, e.g., 256, 512, 768, 1024)
// height: image height (multiple of 16)
// Returns: 0 on success, negative error code on failure
int32_t z_image_generate(
    const char* prompt,
    const char* output_path,
    const char* model_dir,
    int32_t width,
    int32_t height
);

// ============================================================================
// Generation Settings
// ============================================================================

// Set the number of inference steps (4-50, default: 8)
// Z-Image Turbo is optimized for 4-8 steps
// Returns: 0 on success
int32_t z_image_set_num_steps(int32_t steps);

// Get the current number of inference steps
int32_t z_image_get_num_steps(void);

// Set the random seed for reproducible generation
// seed: random seed (0 to disable fixed seed and use random)
// Returns: 0 on success
int32_t z_image_set_seed(uint64_t seed);

// Get the current random seed (0 if using random)
uint64_t z_image_get_seed(void);

// ============================================================================
// Memory Optimization
// ============================================================================

// Set attention slice size for memory optimization
// slice_size: 0 = no slicing (fastest, most memory)
//             1 = process 1 head at a time (slowest, least memory)
//             2-4 = low memory (~12-16GB VRAM)
//             5-8 = medium memory (~16-20GB VRAM)
//             8+ = high memory savings
// Returns: 0 on success
int32_t z_image_set_attention_slice_size(int32_t slice_size);

// Get the current attention slice size
int32_t z_image_get_attention_slice_size(void);

// Enable/disable low memory mode
// When enabled, text encoder is unloaded during diffusion to save ~7.5GB VRAM
// enabled: 1 to enable, 0 to disable
// Returns: 0 on success
int32_t z_image_set_low_memory_mode(int32_t enabled);

// Check if low memory mode is enabled
// Returns: 1 if enabled, 0 if disabled
int32_t z_image_get_low_memory_mode(void);


// ============================================================================
// Text Chat (Qwen3-0.6B)
// ============================================================================

// Initialize the text generation model
// model_dir: path to directory containing chat model files
// Returns: 0 on success, negative error code on failure
int32_t qwen3_init(const char* model_dir);

// Check if text generation model is loaded
// Returns: 1 if model is loaded, 0 otherwise
int32_t qwen3_is_loaded(void);

// Generate text from a prompt
// prompt: user message
// max_tokens: maximum tokens to generate
// temperature: sampling temperature (0.0-1.5)
// Returns: pointer to generated text (caller must free with z_image_free_string)
char* qwen3_generate(const char* prompt, int32_t max_tokens, float temperature);

// Unload text generation model from GPU memory
// Returns: 0 on success
int32_t qwen3_unload(void);

// ============================================================================
// Utility Functions
// ============================================================================

// Get the last error message
// Returns: pointer to error string (valid until next call)
char* z_image_get_error(void);

// Free a string returned by this library
void z_image_free_string(char* s);

#ifdef __cplusplus
}
#endif

#endif // Z_IMAGE_FFI_H
