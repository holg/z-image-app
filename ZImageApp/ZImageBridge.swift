import Foundation

/// Bridge to the Z-Image Rust library via FFI
class ZImageBridge {
    static let shared = ZImageBridge()

    private init() {}

    // MARK: - Image Generation

    /// Load image generation models into memory
    func loadImageModels(modelDir: String) -> Bool {
        let result = z_image_load_models(modelDir)
        return result == 0
    }

    /// Unload image generation models from memory
    func unloadImageModels() -> Bool {
        let result = z_image_unload_models()
        return result == 0
    }

    /// Check if image models are loaded
    var imageModelsLoaded: Bool {
        return z_image_models_loaded()
    }

    /// Generate an image from a text prompt
    func generateImage(
        prompt: String,
        width: Int32,
        height: Int32,
        modelDir: String,
        outputPath: String,
        settings: ZImageGenerationSettings? = nil
    ) -> Bool {
        var settingsValue = settings ?? z_image_default_settings()
        let result = z_image_generate(
            prompt,
            width,
            height,
            modelDir,
            outputPath,
            &settingsValue
        )
        return result == 0
    }

    /// Generate image with cached models
    func generateImageCached(
        prompt: String,
        width: Int32,
        height: Int32,
        outputPath: String,
        settings: ZImageGenerationSettings? = nil
    ) -> Bool {
        var settingsValue = settings ?? z_image_default_settings()
        let result = z_image_generate_cached(
            prompt,
            width,
            height,
            outputPath,
            &settingsValue
        )
        return result == 0
    }

    // MARK: - Chat

    /// Load chat model
    func loadChatModel(modelDir: String) -> Bool {
        let result = z_image_load_chat_model(modelDir)
        return result == 0
    }

    /// Unload chat model
    func unloadChatModel() -> Bool {
        let result = z_image_unload_chat_model()
        return result == 0
    }

    /// Check if chat model is loaded
    var chatModelLoaded: Bool {
        return z_image_chat_model_loaded()
    }

    /// Generate chat response
    func chat(prompt: String, maxTokens: Int32 = 512) -> String? {
        let bufferSize: Int32 = 8192
        var buffer = [CChar](repeating: 0, count: Int(bufferSize))

        let result = z_image_chat(prompt, maxTokens, &buffer, bufferSize)

        if result >= 0 {
            return String(cString: buffer)
        }
        return nil
    }

    // MARK: - Utilities

    /// Get the last error message
    var lastError: String? {
        guard let ptr = z_image_get_last_error() else { return nil }
        return String(cString: ptr)
    }

    /// Get library version
    var version: String {
        guard let ptr = z_image_version() else { return "unknown" }
        return String(cString: ptr)
    }

    /// Create default generation settings
    func defaultSettings() -> ZImageGenerationSettings {
        return z_image_default_settings()
    }
}

// MARK: - Settings Extension

extension ZImageGenerationSettings {
    static var `default`: ZImageGenerationSettings {
        return z_image_default_settings()
    }

    static func custom(
        numInferenceSteps: Int32 = 8,
        seed: Int64 = 0,
        useAttentionSlicing: Bool = false,
        sliceSize: Int32 = 0,
        lowMemoryMode: Bool = false
    ) -> ZImageGenerationSettings {
        return ZImageGenerationSettings(
            num_inference_steps: numInferenceSteps,
            seed: seed,
            use_attention_slicing: useAttentionSlicing,
            slice_size: sliceSize,
            low_memory_mode: lowMemoryMode
        )
    }
}
