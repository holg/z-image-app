import Foundation

// MARK: - C FFI Declarations

// These would normally come from the generated longcat.h header
// For now, declare them manually

@_silgen_name("longcat_init")
func longcat_init() -> Int32

@_silgen_name("longcat_load_models")
func longcat_load_models(_ model_dir: UnsafePointer<CChar>) -> Int32

@_silgen_name("longcat_unload_models")
func longcat_unload_models() -> Int32

@_silgen_name("longcat_models_loaded")
func longcat_models_loaded() -> Int32

@_silgen_name("longcat_generate_video")
func longcat_generate_video(_ prompt: UnsafePointer<CChar>, _ output_path: UnsafePointer<CChar>) -> Int32

@_silgen_name("longcat_generate_video_from_image")
func longcat_generate_video_from_image(_ image_path: UnsafePointer<CChar>, _ prompt: UnsafePointer<CChar>, _ output_path: UnsafePointer<CChar>) -> Int32

@_silgen_name("longcat_set_num_steps")
func longcat_set_num_steps(_ steps: Int32)

@_silgen_name("longcat_get_num_steps")
func longcat_get_num_steps() -> Int32

@_silgen_name("longcat_set_guidance_scale")
func longcat_set_guidance_scale(_ scale: Float)

@_silgen_name("longcat_get_guidance_scale")
func longcat_get_guidance_scale() -> Float

@_silgen_name("longcat_set_num_frames")
func longcat_set_num_frames(_ frames: Int32)

@_silgen_name("longcat_get_num_frames")
func longcat_get_num_frames() -> Int32

@_silgen_name("longcat_set_video_size")
func longcat_set_video_size(_ width: Int32, _ height: Int32)

@_silgen_name("longcat_get_video_width")
func longcat_get_video_width() -> Int32

@_silgen_name("longcat_get_video_height")
func longcat_get_video_height() -> Int32

@_silgen_name("longcat_set_video_fps")
func longcat_set_video_fps(_ fps: Int32)

@_silgen_name("longcat_get_video_fps")
func longcat_get_video_fps() -> Int32

@_silgen_name("longcat_get_error")
func longcat_get_error() -> UnsafeMutablePointer<CChar>?

@_silgen_name("longcat_free_string")
func longcat_free_string(_ s: UnsafeMutablePointer<CChar>?)

// Mode 3: Text-to-Image-to-Video (requires both z-image and longcat models)
@_silgen_name("longcat_generate_video_from_text_with_zimage")
func longcat_generate_video_from_text_with_zimage(_ prompt: UnsafePointer<CChar>, _ output_path: UnsafePointer<CChar>) -> Int32

// Memory optimization
@_silgen_name("longcat_set_attention_slice_size")
func longcat_set_attention_slice_size(_ slice_size: Int32)

@_silgen_name("longcat_get_attention_slice_size")
func longcat_get_attention_slice_size() -> Int32

@_silgen_name("longcat_enable_low_memory_mode")
func longcat_enable_low_memory_mode()

@_silgen_name("longcat_estimate_memory_mb")
func longcat_estimate_memory_mb() -> Int32

// Pause/Resume/Cancel control
@_silgen_name("longcat_pause")
func longcat_pause()

@_silgen_name("longcat_resume")
func longcat_resume()

@_silgen_name("longcat_cancel")
func longcat_cancel()

@_silgen_name("longcat_is_paused")
func longcat_is_paused() -> Int32

@_silgen_name("longcat_is_cancelled")
func longcat_is_cancelled() -> Int32

@_silgen_name("longcat_get_progress")
func longcat_get_progress() -> Float

@_silgen_name("longcat_get_current_step")
func longcat_get_current_step() -> Int32

@_silgen_name("longcat_get_total_steps")
func longcat_get_total_steps() -> Int32

@_silgen_name("longcat_reset_control")
func longcat_reset_control()

// MARK: - Swift Bridge

/// Bridge to the LongCat Rust library for video generation
class LongCatBridge {
    static let shared = LongCatBridge()

    private var initialized = false

    private init() {}

    // MARK: - Initialization

    /// Initialize the LongCat GPU device
    func initialize() {
        guard !initialized else { return }
        longcat_init()
        initialized = true
        print("[LongCat] Device initialized")
    }

    // MARK: - Model Loading

    /// Load LongCat video generation models
    /// - Parameter modelDir: Directory containing the model files
    /// - Returns: true on success
    func loadModels(modelDir: String) -> Bool {
        initialize()
        let result = longcat_load_models(modelDir)
        return result == 0
    }

    /// Unload models from GPU memory
    func unloadModels() -> Bool {
        let result = longcat_unload_models()
        return result == 0
    }

    /// Check if video models are loaded
    var modelsLoaded: Bool {
        return longcat_models_loaded() == 1
    }

    // MARK: - Video Generation

    /// Generate video from text prompt
    /// - Parameters:
    ///   - prompt: Text description of the video
    ///   - outputPath: Path to save the generated video
    /// - Returns: true on success
    func generateVideo(prompt: String, outputPath: String) -> Bool {
        let result = longcat_generate_video(prompt, outputPath)
        return result == 0
    }

    /// Generate video from an existing image (Image-to-Video)
    /// - Parameters:
    ///   - imagePath: Path to the input image
    ///   - prompt: Text description of the video motion
    ///   - outputPath: Path to save the generated video
    /// - Returns: true on success
    func generateVideoFromImage(imagePath: String, prompt: String, outputPath: String) -> Bool {
        let result = longcat_generate_video_from_image(imagePath, prompt, outputPath)
        return result == 0
    }

    /// Generate video by first creating image with z-image (Text-to-Image-to-Video)
    /// REQUIRES both z-image AND longcat models to be loaded
    /// - Parameters:
    ///   - prompt: Text description
    ///   - outputPath: Path to save the generated video
    /// - Returns: true on success
    func generateVideoFromTextWithZImage(prompt: String, outputPath: String) -> Bool {
        let result = longcat_generate_video_from_text_with_zimage(prompt, outputPath)
        return result == 0
    }

    // MARK: - Memory Optimization

    /// Enable low-memory mode with recommended settings
    /// Sets attention slice to 512, disables CFG
    /// Call this BEFORE loading models for best memory savings
    func enableLowMemoryMode() {
        longcat_enable_low_memory_mode()
        print("[LongCat] Low memory mode enabled")
    }

    /// Set attention slice size for memory optimization
    /// - Parameter sliceSize: 0=full attention, 512=recommended for most Macs, 256=very low mem
    func setAttentionSliceSize(_ sliceSize: Int32) {
        longcat_set_attention_slice_size(sliceSize)
    }

    /// Get current attention slice size
    var attentionSliceSize: Int32 {
        return longcat_get_attention_slice_size()
    }

    /// Get estimated memory usage in MB for current settings
    var estimatedMemoryMB: Int32 {
        return longcat_estimate_memory_mb()
    }

    // MARK: - Generation Settings

    /// Set number of inference steps (default: 50)
    func setNumSteps(_ steps: Int32) {
        longcat_set_num_steps(steps)
    }

    /// Get current number of inference steps
    var numSteps: Int32 {
        return longcat_get_num_steps()
    }

    /// Set guidance scale (default: 5.0)
    func setGuidanceScale(_ scale: Float) {
        longcat_set_guidance_scale(scale)
    }

    /// Get current guidance scale
    var guidanceScale: Float {
        return longcat_get_guidance_scale()
    }

    /// Set number of frames to generate (default: 81 = ~5 sec at 15fps)
    func setNumFrames(_ frames: Int32) {
        longcat_set_num_frames(frames)
    }

    /// Get current number of frames
    var numFrames: Int32 {
        return longcat_get_num_frames()
    }

    /// Set video dimensions
    func setVideoSize(width: Int32, height: Int32) {
        longcat_set_video_size(width, height)
    }

    /// Get video width
    var videoWidth: Int32 {
        return longcat_get_video_width()
    }

    /// Get video height
    var videoHeight: Int32 {
        return longcat_get_video_height()
    }

    /// Set video FPS (default: 15)
    func setVideoFPS(_ fps: Int32) {
        longcat_set_video_fps(fps)
    }

    /// Get video FPS
    var videoFPS: Int32 {
        return longcat_get_video_fps()
    }

    // MARK: - Error Handling

    /// Get the last error message
    var lastError: String? {
        guard let ptr = longcat_get_error() else { return nil }
        let error = String(cString: ptr)
        longcat_free_string(ptr)
        return error
    }

    // MARK: - Generation Control (Pause/Resume/Cancel)

    /// Pause the current video generation
    /// The generation will pause after completing the current step.
    func pause() {
        longcat_pause()
        print("[LongCat] Pause requested")
    }

    /// Resume a paused video generation
    func resume() {
        longcat_resume()
        print("[LongCat] Resume requested")
    }

    /// Cancel the current video generation
    /// The generation will be cancelled at the end of the current step.
    func cancel() {
        longcat_cancel()
        print("[LongCat] Cancel requested")
    }

    /// Check if generation is currently paused
    var isPaused: Bool {
        return longcat_is_paused() == 1
    }

    /// Check if generation was cancelled
    var isCancelled: Bool {
        return longcat_is_cancelled() == 1
    }

    /// Get the current generation progress as a percentage (0.0 - 100.0)
    var progress: Float {
        return longcat_get_progress()
    }

    /// Get the current step number
    var currentStep: Int32 {
        return longcat_get_current_step()
    }

    /// Get the total number of steps
    var totalSteps: Int32 {
        return longcat_get_total_steps()
    }

    /// Reset the generation control for a new generation
    /// Call this before starting a new generation to clear any previous cancel state.
    func resetControl() {
        longcat_reset_control()
    }
}

// MARK: - Video Generation Presets

enum VideoPreset: String, CaseIterable {
    case fast480p = "Fast 480p"
    case standard480p = "Standard 480p"
    case quality720p = "Quality 720p"

    var description: String {
        switch self {
        case .fast480p: return "480p, 2 sec, 25 steps"
        case .standard480p: return "480p, 5 sec, 50 steps"
        case .quality720p: return "720p, 5 sec, 50 steps"
        }
    }

    var width: Int32 {
        switch self {
        case .fast480p, .standard480p: return 832
        case .quality720p: return 1280
        }
    }

    var height: Int32 {
        switch self {
        case .fast480p, .standard480p: return 480
        case .quality720p: return 720
        }
    }

    var numFrames: Int32 {
        switch self {
        case .fast480p: return 41  // ~2.7 sec at 15fps
        case .standard480p, .quality720p: return 81  // ~5.4 sec at 15fps
        }
    }

    var numSteps: Int32 {
        switch self {
        case .fast480p: return 25
        case .standard480p, .quality720p: return 50
        }
    }

    var fps: Int32 {
        switch self {
        case .fast480p, .standard480p: return 15
        case .quality720p: return 30
        }
    }

    func apply() {
        let bridge = LongCatBridge.shared
        bridge.setVideoSize(width: width, height: height)
        bridge.setNumFrames(numFrames)
        bridge.setNumSteps(numSteps)
        bridge.setVideoFPS(fps)
    }
}
