import SwiftUI
import AVKit
import AVFoundation
import UniformTypeIdentifiers

// MARK: - Safe Video Player Wrapper
struct SafeVideoPlayer: NSViewRepresentable {
    let url: URL

    func makeNSView(context: Context) -> AVPlayerView {
        let playerView = AVPlayerView()
        playerView.controlsStyle = .floating
        playerView.showsFullScreenToggleButton = true
        let player = AVPlayer(url: url)
        playerView.player = player
        return playerView
    }

    func updateNSView(_ nsView: AVPlayerView, context: Context) {
        // Only update if URL changed
        if let currentItem = nsView.player?.currentItem,
           let currentURL = (currentItem.asset as? AVURLAsset)?.url,
           currentURL != url {
            nsView.player = AVPlayer(url: url)
        }
    }
}

/// Video Generation View - combines ZImage for initial frame + LongCat for I2V
struct VideoGenerationView: View {
    @Binding var modelDirectory: String
    @Binding var imageModelsLoaded: Bool
    @ObservedObject var logManager: LogManager

    // Video generation state
    @State private var videoModelsLoaded = false
    @State private var isGenerating = false
    @State private var generationProgress: Double = 0
    @State private var statusMessage = "Load models to begin"

    // Input settings
    @State private var prompt: String = ""
    @State private var selectedPreset: VideoPreset = .fast480p
    @State private var numFrames: Double = 41
    @State private var numSteps: Double = 25
    @State private var guidanceScale: Double = 5.0
    @State private var videoWidth: Double = 832
    @State private var videoHeight: Double = 480
    @State private var videoFPS: Double = 15

    // Generation mode
    @State private var generationMode: GenerationMode = .textToVideo

    // Source image for I2V
    @State private var sourceImage: NSImage? = nil
    @State private var sourceImagePath: String? = nil

    // Generated content
    @State private var generatedVideoURL: URL? = nil
    @State private var generatedFirstFrame: NSImage? = nil

    // Memory settings
    @State private var lowMemoryMode = true
    @State private var attentionSliceSize: Double = 512
    @State private var estimatedMemoryMB: Int32 = 0

    enum GenerationMode: String, CaseIterable {
        case textToVideo = "Text to Video (T2V)"
        case imageToVideo = "Image to Video (I2V)"
        case textToImageToVideo = "Text → Image → Video"

        var icon: String {
            switch self {
            case .textToVideo: return "text.bubble"
            case .imageToVideo: return "photo.on.rectangle"
            case .textToImageToVideo: return "wand.and.stars"
            }
        }

        var description: String {
            switch self {
            case .textToVideo: return "Pure text-to-video (LongCat only)"
            case .imageToVideo: return "Animate your image"
            case .textToImageToVideo: return "Z-Image + LongCat pipeline"
            }
        }

        var requiresImageModels: Bool {
            switch self {
            case .textToVideo: return false
            case .imageToVideo: return false
            case .textToImageToVideo: return true
            }
        }
    }

    var body: some View {
        HStack(spacing: 0) {
            // Left panel - Controls (wider)
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    // Model status
                    modelStatusSection

                    Divider()

                    // Generation mode picker
                    generationModeSection

                    // Prompt input
                    promptSection

                    // Source image (for I2V mode)
                    if generationMode == .imageToVideo {
                        sourceImageSection
                    }

                    // Video settings
                    videoSettingsSection

                    // Generation settings
                    generationSettingsSection

                    // Memory settings
                    memorySettingsSection

                    // Generate button
                    generateButton

                    // Status
                    Text(statusMessage)
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .lineLimit(3)
                }
                .padding(16)
            }
            .frame(minWidth: 420, maxWidth: 500)
            .background(Color(NSColor.windowBackgroundColor))

            Divider()

            // Right panel - Preview (constrained)
            previewPanel
                .frame(minWidth: 300, maxWidth: 500)
        }
    }

    // MARK: - Model Status Section

    private var modelStatusSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Image models (only needed for T2I2V mode)
            HStack {
                Circle()
                    .fill(imageModelsLoaded ? Color.green : (generationMode.requiresImageModels ? Color.red : Color.gray))
                    .frame(width: 10, height: 10)
                Text("Image Models (Z-Image)")
                    .font(.caption)
                Spacer()
                if !imageModelsLoaded && generationMode.requiresImageModels {
                    Text("Required")
                        .font(.caption2)
                        .foregroundColor(.red)
                } else if !imageModelsLoaded {
                    Text("Not needed")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            }

            // Video models (always needed)
            HStack {
                Circle()
                    .fill(videoModelsLoaded ? Color.green : Color.orange)
                    .frame(width: 10, height: 10)
                Text("Video Models (LongCat)")
                    .font(.caption)
                Spacer()
                if !videoModelsLoaded && !modelDirectory.isEmpty {
                    Button("Load") {
                        loadVideoModels()
                    }
                    .font(.caption)
                    .disabled(isGenerating)
                }
            }

            // Mode-specific hints
            if generationMode.requiresImageModels && !imageModelsLoaded {
                Text("⚠️ T2I2V mode requires Z-Image models")
                    .font(.caption2)
                    .foregroundColor(.orange)
            }

            // Memory estimate
            if videoModelsLoaded {
                HStack {
                    Text("Est. Memory:")
                        .font(.caption)
                    Text("\(estimatedMemoryMB) MB (\(String(format: "%.1f", Float(estimatedMemoryMB) / 1000)) GB)")
                        .font(.caption)
                        .foregroundColor(estimatedMemoryMB > 32000 ? .red : .secondary)
                }
            }
        }
    }

    // MARK: - Generation Mode Section

    private var generationModeSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Mode")
                .font(.headline)

            Picker("", selection: $generationMode) {
                ForEach(GenerationMode.allCases, id: \.self) { mode in
                    Label(mode.rawValue, systemImage: mode.icon)
                        .tag(mode)
                }
            }
            .pickerStyle(.segmented)

            Text(generationMode.description)
                .font(.caption2)
                .foregroundColor(.secondary)
        }
    }

    // MARK: - Prompt Section

    private var promptSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Prompt")
                .font(.headline)

            TextEditor(text: $prompt)
                .frame(height: 70)
                .font(.body)
                .padding(4)
                .background(Color(NSColor.textBackgroundColor))
                .cornerRadius(8)

            if generationMode == .textToVideo {
                Text("Describe the video content. First frame will be generated with ZImage.")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            } else {
                Text("Describe the motion/action for the video.")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
        }
    }

    // MARK: - Source Image Section (I2V)

    private var sourceImageSection: some View {
        GroupBox("Source Image") {
            VStack(spacing: 8) {
                if let image = sourceImage {
                    Image(nsImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(maxHeight: 150)
                        .cornerRadius(8)

                    HStack {
                        Button("Change") {
                            selectSourceImage()
                        }
                        Button("Clear") {
                            sourceImage = nil
                            sourceImagePath = nil
                        }
                    }
                    .font(.caption)
                } else {
                    Button(action: selectSourceImage) {
                        VStack(spacing: 8) {
                            Image(systemName: "photo.badge.plus")
                                .font(.largeTitle)
                            Text("Select Image")
                        }
                        .frame(maxWidth: .infinity)
                        .frame(height: 100)
                        .background(Color(NSColor.controlBackgroundColor))
                        .cornerRadius(8)
                    }
                    .buttonStyle(.plain)

                    Text("Or generate first frame with ZImage")
                        .font(.caption2)
                        .foregroundColor(.secondary)

                    Button("Generate from Prompt") {
                        generateFirstFrame()
                    }
                    .disabled(!imageModelsLoaded || prompt.isEmpty || isGenerating)
                }
            }
            .padding(4)
        }
    }

    // MARK: - Video Settings Section

    private var videoSettingsSection: some View {
        GroupBox("Video Settings") {
            VStack(alignment: .leading, spacing: 8) {
                // Presets
                HStack(spacing: 4) {
                    ForEach(VideoPreset.allCases, id: \.self) { preset in
                        Button(preset.rawValue.replacingOccurrences(of: " ", with: "\n")) {
                            selectedPreset = preset
                            applyPreset(preset)
                        }
                        .buttonStyle(.bordered)
                        .tint(selectedPreset == preset ? .accentColor : .secondary)
                        .font(.caption2)
                    }
                }

                Divider()

                // Size
                HStack {
                    Text("Size:")
                        .frame(width: 50, alignment: .leading)
                    Text("\(Int(videoWidth))x\(Int(videoHeight))")
                        .monospacedDigit()
                    Spacer()
                }

                // Frames
                HStack {
                    Text("Frames:")
                        .frame(width: 50, alignment: .leading)
                    Slider(value: $numFrames, in: 17...161, step: 8)
                    Text("\(Int(numFrames))")
                        .frame(width: 35)
                        .monospacedDigit()
                }

                // Duration display
                HStack {
                    Text("Duration:")
                        .frame(width: 50, alignment: .leading)
                    Text(String(format: "%.1f sec @ %d fps", Double(numFrames) / videoFPS, Int(videoFPS)))
                        .foregroundColor(.secondary)
                    Spacer()
                }
            }
            .padding(4)
        }
    }

    // MARK: - Generation Settings Section

    private var generationSettingsSection: some View {
        GroupBox("Generation") {
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text("Steps:")
                    Slider(value: $numSteps, in: 10...100, step: 5)
                    Text("\(Int(numSteps))")
                        .frame(width: 30)
                        .monospacedDigit()
                }

                HStack {
                    Text("Guidance:")
                    Slider(value: $guidanceScale, in: 1...15, step: 0.5)
                    Text(String(format: "%.1f", guidanceScale))
                        .frame(width: 35)
                        .monospacedDigit()
                }
            }
            .padding(4)
        }
    }

    // MARK: - Memory Settings Section

    private var memorySettingsSection: some View {
        GroupBox("Memory") {
            VStack(alignment: .leading, spacing: 6) {
                Toggle("Low Memory Mode", isOn: $lowMemoryMode)
                    .onChange(of: lowMemoryMode) { newValue in
                        if newValue {
                            LongCatBridge.shared.enableLowMemoryMode()
                            attentionSliceSize = 512
                            guidanceScale = 1.0 // CFG disabled
                        } else {
                            LongCatBridge.shared.setAttentionSliceSize(0)
                            attentionSliceSize = 0
                        }
                        updateMemoryEstimate()
                    }

                if !lowMemoryMode {
                    HStack {
                        Text("Attn Slice:")
                        Slider(value: $attentionSliceSize, in: 0...2048, step: 256) { editing in
                            // Only apply when done editing (finger lifted)
                            if !editing {
                                LongCatBridge.shared.setAttentionSliceSize(Int32(attentionSliceSize))
                                updateMemoryEstimate()
                            }
                        }
                        Text(attentionSliceSize == 0 ? "Full" : "\(Int(attentionSliceSize))")
                            .frame(width: 45)
                            .monospacedDigit()
                    }
                }

                if lowMemoryMode {
                    Text("✓ Attention slice: 512, CFG disabled")
                        .font(.caption2)
                        .foregroundColor(.green)
                }
            }
            .padding(4)
        }
    }

    // MARK: - Generate Button

    private var generateButton: some View {
        VStack(spacing: 8) {
            if isGenerating {
                // Progress section during generation
                VStack(spacing: 4) {
                    HStack {
                        Text("Step \(currentStep)/\(totalSteps)")
                            .font(.caption)
                            .monospacedDigit()
                        Spacer()
                        Text(String(format: "%.0f%%", generationProgress))
                            .font(.caption)
                            .monospacedDigit()
                    }

                    ProgressView(value: generationProgress / 100.0)

                    HStack(spacing: 12) {
                        // Pause/Resume button
                        Button(action: togglePause) {
                            Label(isPaused ? "Resume" : "Pause",
                                  systemImage: isPaused ? "play.fill" : "pause.fill")
                        }
                        .buttonStyle(.bordered)

                        // Cancel button
                        Button(action: cancelGeneration) {
                            Label("Cancel", systemImage: "xmark.circle.fill")
                        }
                        .buttonStyle(.bordered)
                        .tint(.red)
                    }

                    if isPaused {
                        Text("Generation paused")
                            .font(.caption)
                            .foregroundColor(.orange)
                    }
                }
            } else {
                // Generate button when not generating
                Button(action: generateVideo) {
                    HStack {
                        Image(systemName: "film")
                        Text("Generate Video")
                    }
                    .frame(maxWidth: .infinity)
                    .padding(8)
                }
                .buttonStyle(.borderedProminent)
                .disabled(!canGenerate)
            }
        }
    }

    @State private var isPaused = false
    @State private var currentStep: Int32 = 0
    @State private var totalSteps: Int32 = 0
    @State private var progressTimer: Timer? = nil

    private func togglePause() {
        if isPaused {
            LongCatBridge.shared.resume()
        } else {
            LongCatBridge.shared.pause()
        }
        isPaused.toggle()
    }

    private func cancelGeneration() {
        LongCatBridge.shared.cancel()
        statusMessage = "Cancelling..."
    }

    private func startProgressTimer() {
        progressTimer?.invalidate()
        progressTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { _ in
            DispatchQueue.main.async {
                self.generationProgress = Double(LongCatBridge.shared.progress)
                self.currentStep = LongCatBridge.shared.currentStep
                self.totalSteps = LongCatBridge.shared.totalSteps
                self.isPaused = LongCatBridge.shared.isPaused
            }
        }
    }

    private func stopProgressTimer() {
        progressTimer?.invalidate()
        progressTimer = nil
    }

    private var canGenerate: Bool {
        guard videoModelsLoaded && !prompt.isEmpty else { return false }

        switch generationMode {
        case .textToVideo:
            return true // Only needs video models
        case .imageToVideo:
            return sourceImage != nil // Needs source image
        case .textToImageToVideo:
            return imageModelsLoaded // Needs both models
        }
    }

    private func updateMemoryEstimate() {
        estimatedMemoryMB = LongCatBridge.shared.estimatedMemoryMB
    }

    // MARK: - Preview Panel

    private var previewPanel: some View {
        VStack {
            if let videoURL = generatedVideoURL {
                // Video player (using safe wrapper to avoid SwiftUI/AVKit crash)
                SafeVideoPlayer(url: videoURL)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)

                HStack {
                    Button("Save Video") {
                        saveVideo()
                    }
                    Button("Open in Finder") {
                        NSWorkspace.shared.selectFile(videoURL.path, inFileViewerRootedAtPath: "")
                    }
                }
                .padding(.bottom)
            } else if let firstFrame = generatedFirstFrame {
                // Show first frame preview during generation
                VStack {
                    Text("First Frame Preview")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    Image(nsImage: firstFrame)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .cornerRadius(8)

                    if isGenerating {
                        ProgressView(value: generationProgress)
                            .padding(.horizontal)
                        Text("Generating video frames...")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
                .padding()
            } else {
                VStack {
                    Image(systemName: "film")
                        .font(.system(size: 48))
                        .foregroundColor(.secondary)
                    Text("Generated video will appear here")
                        .foregroundColor(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .frame(minWidth: 400)
    }

    // MARK: - Actions

    private func loadVideoModels() {
        statusMessage = "Loading video models..."
        logManager.log("[LongCat] Loading models from \(modelDirectory)...")

        // Apply memory settings BEFORE loading
        if lowMemoryMode {
            LongCatBridge.shared.enableLowMemoryMode()
            logManager.log("[LongCat] Low memory mode enabled")
        }

        DispatchQueue.global(qos: .userInitiated).async {
            let success = LongCatBridge.shared.loadModels(modelDir: modelDirectory)

            DispatchQueue.main.async {
                videoModelsLoaded = success
                if success {
                    statusMessage = "Video models loaded"
                    logManager.log("[LongCat] Models loaded successfully")
                    updateMemoryEstimate()
                } else {
                    let error = LongCatBridge.shared.lastError ?? "Unknown error"
                    statusMessage = "Failed to load models: \(error)"
                    logManager.log("[LongCat] Error: \(error)")
                }
            }
        }
    }

    private func applyPreset(_ preset: VideoPreset) {
        numFrames = Double(preset.numFrames)
        numSteps = Double(preset.numSteps)
        videoWidth = Double(preset.width)
        videoHeight = Double(preset.height)
        videoFPS = Double(preset.fps)
    }

    private func selectSourceImage() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [UTType.png, UTType.jpeg, UTType.heic]
        panel.allowsMultipleSelection = false

        if panel.runModal() == .OK, let url = panel.url {
            sourceImagePath = url.path
            sourceImage = NSImage(contentsOf: url)
        }
    }

    private func generateFirstFrame() {
        guard imageModelsLoaded, !prompt.isEmpty else { return }

        isGenerating = true
        statusMessage = "Generating first frame with ZImage..."
        logManager.log("[ZImage] Generating first frame for I2V...")

        let tempPath = FileManager.default.temporaryDirectory
            .appendingPathComponent("longcat_first_frame_\(UUID().uuidString).png")
            .path

        DispatchQueue.global(qos: .userInitiated).async {
            // Use ZImage to generate the first frame
            ZImageBridge.shared.setNumSteps(8) // Quick generation
            let success = ZImageBridge.shared.generateImage(
                prompt: prompt,
                width: Int32(videoWidth),
                height: Int32(videoHeight),
                modelDir: modelDirectory,
                outputPath: tempPath
            )

            DispatchQueue.main.async {
                isGenerating = false

                if success, let image = NSImage(contentsOfFile: tempPath) {
                    sourceImage = image
                    sourceImagePath = tempPath
                    generatedFirstFrame = image
                    statusMessage = "First frame generated"
                    logManager.log("[ZImage] First frame generated successfully")
                } else {
                    let error = ZImageBridge.shared.lastError ?? "Unknown error"
                    statusMessage = "Failed to generate first frame: \(error)"
                    logManager.log("[ZImage] Error: \(error)")
                }
            }
        }
    }

    private func generateVideo() {
        isGenerating = true
        generationProgress = 0
        generatedVideoURL = nil
        isPaused = false
        currentStep = 0
        totalSteps = Int32(numSteps)

        // Reset generation control for new generation
        LongCatBridge.shared.resetControl()

        // Start progress monitoring
        startProgressTimer()

        // Apply current settings
        LongCatBridge.shared.setVideoSize(width: Int32(videoWidth), height: Int32(videoHeight))
        LongCatBridge.shared.setNumFrames(Int32(numFrames))
        LongCatBridge.shared.setNumSteps(Int32(numSteps))
        LongCatBridge.shared.setGuidanceScale(Float(guidanceScale))
        LongCatBridge.shared.setVideoFPS(Int32(videoFPS))

        let tempVideoPath = FileManager.default.temporaryDirectory
            .appendingPathComponent("longcat_video_\(UUID().uuidString).mp4")

        switch generationMode {
        case .textToVideo:
            // Mode 1: Pure T2V - uses LongCat's text encoder directly
            statusMessage = "Generating video from text..."
            logManager.log("[LongCat] T2V mode: \"\(prompt)\"")

            DispatchQueue.global(qos: .userInitiated).async {
                let success = LongCatBridge.shared.generateVideo(
                    prompt: prompt,
                    outputPath: tempVideoPath.path
                )

                DispatchQueue.main.async {
                    isGenerating = false
                    stopProgressTimer()
                    generationProgress = 100.0

                    if success {
                        generatedVideoURL = tempVideoPath
                        statusMessage = "Video generated successfully!"
                        logManager.log("[LongCat] Video generated: \(tempVideoPath.path)")
                    } else if LongCatBridge.shared.isCancelled {
                        statusMessage = "Generation cancelled"
                        logManager.log("[LongCat] Generation cancelled by user")
                    } else {
                        let error = LongCatBridge.shared.lastError ?? "Unknown error"
                        statusMessage = "Failed to generate video: \(error)"
                        logManager.log("[LongCat] Error: \(error)")
                    }
                }
            }

        case .imageToVideo:
            // Mode 2: I2V - animate user's image
            guard let imagePath = sourceImagePath else {
                isGenerating = false
                statusMessage = "No source image selected"
                return
            }

            statusMessage = "Generating video from image..."
            logManager.log("[LongCat] I2V mode: \(imagePath)")

            DispatchQueue.global(qos: .userInitiated).async {
                let success = LongCatBridge.shared.generateVideoFromImage(
                    imagePath: imagePath,
                    prompt: prompt,
                    outputPath: tempVideoPath.path
                )

                DispatchQueue.main.async {
                    isGenerating = false
                    stopProgressTimer()
                    generationProgress = 100.0

                    if success {
                        generatedVideoURL = tempVideoPath
                        statusMessage = "Video generated successfully!"
                        logManager.log("[LongCat] Video generated: \(tempVideoPath.path)")
                    } else if LongCatBridge.shared.isCancelled {
                        statusMessage = "Generation cancelled"
                        logManager.log("[LongCat] Generation cancelled by user")
                    } else {
                        let error = LongCatBridge.shared.lastError ?? "Unknown error"
                        statusMessage = "Failed to generate video: \(error)"
                        logManager.log("[LongCat] Error: \(error)")
                    }
                }
            }

        case .textToImageToVideo:
            // Mode 3: T2I2V - Z-Image generates frame, then LongCat animates
            statusMessage = "Step 1/2: Generating first frame with Z-Image..."
            logManager.log("[LongCat] T2I2V mode: \"\(prompt)\"")

            DispatchQueue.global(qos: .userInitiated).async {
                let success = LongCatBridge.shared.generateVideoFromTextWithZImage(
                    prompt: prompt,
                    outputPath: tempVideoPath.path
                )

                DispatchQueue.main.async {
                    isGenerating = false
                    stopProgressTimer()
                    generationProgress = 100.0

                    if success {
                        generatedVideoURL = tempVideoPath
                        statusMessage = "Video generated successfully!"
                        logManager.log("[LongCat] Video generated: \(tempVideoPath.path)")
                    } else if LongCatBridge.shared.isCancelled {
                        statusMessage = "Generation cancelled"
                        logManager.log("[LongCat] Generation cancelled by user")
                    } else {
                        let error = LongCatBridge.shared.lastError ?? "Unknown error"
                        statusMessage = "Failed to generate video: \(error)"
                        logManager.log("[LongCat] Error: \(error)")
                    }
                }
            }
        }
    }

    private func saveVideo() {
        guard let videoURL = generatedVideoURL else { return }

        let panel = NSSavePanel()
        panel.allowedContentTypes = [UTType.mpeg4Movie]
        panel.nameFieldStringValue = "generated_video.mp4"

        if panel.runModal() == .OK, let url = panel.url {
            try? FileManager.default.copyItem(at: videoURL, to: url)
        }
    }
}
