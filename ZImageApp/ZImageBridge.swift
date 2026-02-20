import Foundation

/// Bridge to the Z-Image Rust library via FFI
class ZImageBridge {
    static let shared = ZImageBridge()

    private init() {
        // Initialize the GPU device on first access
        z_image_init()
    }

    // MARK: - Image Generation Models

    /// Load image generation models into GPU memory
    /// Returns nil on success, or an error message on failure
    func loadImageModels(modelDir: String) -> String? {
        let result = z_image_load_models(modelDir)
        if result == 0 {
            return nil // Success
        } else {
            // Get the error message
            return lastError ?? "Unknown error loading models"
        }
    }

    /// Unload image generation models from GPU memory
    func unloadImageModels() -> Bool {
        let result = z_image_unload_models()
        return result == 0
    }

    /// Check if image models are loaded
    var imageModelsLoaded: Bool {
        return z_image_models_loaded() == 1
    }

    // MARK: - Image Generation

    /// Generate an image from a text prompt
    func generateImage(
        prompt: String,
        width: Int32,
        height: Int32,
        modelDir: String,
        outputPath: String
    ) -> Bool {
        let result = z_image_generate(prompt, outputPath, modelDir, width, height)
        return result == 0
    }

    // MARK: - Generation Settings

    /// Set the number of inference steps (4-50, default: 8)
    func setNumSteps(_ steps: Int32) {
        z_image_set_num_steps(steps)
    }

    /// Get the current number of inference steps
    var numSteps: Int32 {
        return z_image_get_num_steps()
    }

    /// Set the random seed (0 for random)
    func setSeed(_ seed: UInt64) {
        z_image_set_seed(seed)
    }

    /// Get the current random seed (0 if random)
    var seed: UInt64 {
        return z_image_get_seed()
    }

    // MARK: - Memory Optimization

    /// Set attention slice size for memory optimization
    /// - 0: No slicing (fastest, most memory)
    /// - 1: Slowest, minimum memory
    /// - 2-4: Low memory (~12-16GB)
    /// - 5-8: Medium memory (~16-20GB)
    func setAttentionSliceSize(_ size: Int32) {
        z_image_set_attention_slice_size(size)
    }

    /// Get the current attention slice size
    var attentionSliceSize: Int32 {
        return z_image_get_attention_slice_size()
    }

    /// Enable/disable low memory mode (unloads text encoder during diffusion)
    func setLowMemoryMode(_ enabled: Bool) {
        z_image_set_low_memory_mode(enabled ? 1 : 0)
    }

    /// Check if low memory mode is enabled
    var lowMemoryMode: Bool {
        return z_image_get_low_memory_mode() == 1
    }

    // MARK: - Memory Presets

    /// Apply high VRAM preset (24GB+)
    func applyHighVRAMPreset() {
        z_image_set_attention_slice_size(0)
        z_image_set_low_memory_mode(0)
    }

    /// Apply medium VRAM preset (16GB)
    func applyMediumVRAMPreset() {
        z_image_set_attention_slice_size(8)
        z_image_set_low_memory_mode(0)
    }

    /// Apply low VRAM preset (12GB)
    func applyLowVRAMPreset() {
        z_image_set_attention_slice_size(4)
        z_image_set_low_memory_mode(1)
    }

    /// Apply very low VRAM preset (8GB)
    func applyVeryLowVRAMPreset() {
        z_image_set_attention_slice_size(2)
        z_image_set_low_memory_mode(1)
    }

    // MARK: - Text Chat (Qwen3-0.6B)

    /// Load chat model
    func loadChatModel(modelDir: String) -> Bool {
        let result = qwen3_init(modelDir)
        return result == 0
    }

    /// Unload chat model
    func unloadChatModel() -> Bool {
        let result = qwen3_unload()
        return result == 0
    }

    /// Check if chat model is loaded
    var chatModelLoaded: Bool {
        return qwen3_is_loaded() == 1
    }

    /// Generate chat response
    func chat(prompt: String, maxTokens: Int32 = 512, temperature: Float = 0.7) -> String? {
        guard let ptr = qwen3_generate(prompt, maxTokens, temperature) else {
            return nil
        }
        let result = String(cString: ptr)
        z_image_free_string(ptr)
        return result
    }

    // MARK: - Utilities

    /// Get the last error message
    var lastError: String? {
        guard let ptr = z_image_get_error() else { return nil }
        let error = String(cString: ptr)
        z_image_free_string(ptr)
        return error
    }
}

// MARK: - Memory Preset Enum

enum VRAMPreset: String, CaseIterable {
    case fast = "Fast (no optimization)"
    case balanced = "Balanced"
    case lowPeak = "Lower Peak"
    case minPeak = "Minimum Peak"

    var shortName: String {
        switch self {
        case .fast: return "Fast"
        case .balanced: return "Balanced"
        case .lowPeak: return "Low Peak"
        case .minPeak: return "Min Peak"
        }
    }

    var attentionSliceSize: Int32 {
        switch self {
        case .fast: return 0
        case .balanced: return 8
        case .lowPeak: return 4
        case .minPeak: return 2
        }
    }

    var lowMemoryMode: Bool {
        switch self {
        case .fast, .balanced: return false
        case .lowPeak, .minPeak: return true
        }
    }

    func apply() {
        switch self {
        case .fast: ZImageBridge.shared.applyHighVRAMPreset()
        case .balanced: ZImageBridge.shared.applyMediumVRAMPreset()
        case .lowPeak: ZImageBridge.shared.applyLowVRAMPreset()
        case .minPeak: ZImageBridge.shared.applyVeryLowVRAMPreset()
        }
    }
}
