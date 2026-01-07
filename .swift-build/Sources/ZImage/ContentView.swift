import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @State private var selectedTab = 0
    @State private var modelDirectory: String = ""
    @State private var modelsLoaded = false
    @State private var isLoading = false
    @State private var statusMessage = "Select model directory to begin"

    var body: some View {
        VStack(spacing: 0) {
            // Tab bar
            HStack(spacing: 0) {
                TabButton(title: "Image", icon: "photo", isSelected: selectedTab == 0) {
                    selectedTab = 0
                }
                TabButton(title: "Chat", icon: "bubble.left", isSelected: selectedTab == 1) {
                    selectedTab = 1
                }
                TabButton(title: "Settings", icon: "gear", isSelected: selectedTab == 2) {
                    selectedTab = 2
                }
            }
            .padding(.horizontal)
            .padding(.top, 8)

            Divider()
                .padding(.top, 8)

            // Content
            Group {
                switch selectedTab {
                case 0:
                    ImageGenerationView(
                        modelDirectory: $modelDirectory,
                        modelsLoaded: $modelsLoaded,
                        isLoading: $isLoading,
                        statusMessage: $statusMessage
                    )
                case 1:
                    ChatView(modelDirectory: $modelDirectory)
                case 2:
                    SettingsView(
                        modelDirectory: $modelDirectory,
                        modelsLoaded: $modelsLoaded,
                        isLoading: $isLoading,
                        statusMessage: $statusMessage
                    )
                default:
                    Text("Unknown tab")
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .frame(minWidth: 900, minHeight: 700)
    }
}

// MARK: - Tab Button

struct TabButton: View {
    let title: String
    let icon: String
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.system(size: 20))
                Text(title)
                    .font(.caption)
            }
            .frame(width: 80, height: 50)
            .background(isSelected ? Color.accentColor.opacity(0.2) : Color.clear)
            .cornerRadius(8)
        }
        .buttonStyle(.plain)
        .foregroundColor(isSelected ? .accentColor : .secondary)
    }
}

// MARK: - Image Generation View

struct ImageGenerationView: View {
    @Binding var modelDirectory: String
    @Binding var modelsLoaded: Bool
    @Binding var isLoading: Bool
    @Binding var statusMessage: String

    @State private var prompt = ""
    @State private var width: Double = 512
    @State private var height: Double = 512
    @State private var numSteps: Double = 8
    @State private var seed: String = "0"
    @State private var useSeed = false
    @State private var generatedImage: NSImage?
    @State private var isGenerating = false

    // Memory settings
    @State private var selectedPreset: VRAMPreset = .high
    @State private var attentionSliceSize: Double = 0
    @State private var lowMemoryMode = false

    var body: some View {
        HSplitView {
            // Left panel - Controls
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    // Model status
                    HStack {
                        Circle()
                            .fill(modelsLoaded ? Color.green : Color.red)
                            .frame(width: 10, height: 10)
                        Text(modelsLoaded ? "Models Loaded" : "Models Not Loaded")
                            .font(.caption)
                        Spacer()
                        if !modelsLoaded && !modelDirectory.isEmpty {
                            Button("Load Models") {
                                loadModels()
                            }
                            .disabled(isLoading)
                        }
                    }

                    // Prompt input
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Prompt")
                            .font(.headline)
                        TextEditor(text: $prompt)
                            .frame(height: 80)
                            .font(.body)
                            .padding(4)
                            .background(Color(NSColor.textBackgroundColor))
                            .cornerRadius(8)
                    }

                    // Size controls
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Image Size")
                            .font(.headline)

                        HStack {
                            Text("Width:")
                            Slider(value: $width, in: 256...1024, step: 64)
                            Text("\(Int(width))")
                                .frame(width: 50)
                        }

                        HStack {
                            Text("Height:")
                            Slider(value: $height, in: 256...1024, step: 64)
                            Text("\(Int(height))")
                                .frame(width: 50)
                        }

                        // Size presets
                        HStack {
                            Text("Presets:")
                                .font(.caption)
                            Button("256") { width = 256; height = 256 }
                            Button("512") { width = 512; height = 512 }
                            Button("768") { width = 768; height = 768 }
                            Button("1024") { width = 1024; height = 1024 }
                        }
                        .font(.caption)
                    }

                    // Generation settings - Always visible
                    GroupBox("Generation Options") {
                        VStack(alignment: .leading, spacing: 10) {
                            HStack {
                                Text("Inference Steps:")
                                Slider(value: $numSteps, in: 4...20, step: 1)
                                Text("\(Int(numSteps))")
                                    .frame(width: 30)
                                    .monospacedDigit()
                            }
                            Text("Z-Image Turbo is optimized for 4-8 steps")
                                .font(.caption)
                                .foregroundColor(.secondary)

                            Divider()

                            HStack {
                                Toggle("Fixed Seed:", isOn: $useSeed)
                                    .toggleStyle(.checkbox)
                                TextField("Seed", text: $seed)
                                    .textFieldStyle(.roundedBorder)
                                    .frame(width: 100)
                                    .disabled(!useSeed)
                                if useSeed {
                                    Button("Random") {
                                        seed = String(UInt64.random(in: 1...UInt64.max))
                                    }
                                }
                            }
                        }
                        .padding(8)
                    }

                    // Memory optimization - Always visible
                    GroupBox("Memory Optimization") {
                        VStack(alignment: .leading, spacing: 10) {
                            // VRAM Presets
                            Text("VRAM Preset")
                                .font(.subheadline)
                            Picker("", selection: $selectedPreset) {
                                ForEach(VRAMPreset.allCases, id: \.self) { preset in
                                    Text(preset.rawValue).tag(preset)
                                }
                            }
                            .pickerStyle(.segmented)
                            .onChange(of: selectedPreset) { _, newValue in
                                applyPreset(newValue)
                            }

                            Divider()

                            // Attention slicing
                            HStack {
                                Text("Attention Slicing:")
                                Slider(value: $attentionSliceSize, in: 0...30, step: 1)
                                Text("\(Int(attentionSliceSize))")
                                    .frame(width: 30)
                                    .monospacedDigit()
                            }
                            Text(attentionSliceDescription)
                                .font(.caption)
                                .foregroundColor(.secondary)

                            // Low memory mode
                            Toggle("Low Memory Mode (saves ~7.5GB VRAM)", isOn: $lowMemoryMode)
                                .toggleStyle(.checkbox)
                                .onChange(of: lowMemoryMode) { _, newValue in
                                    ZImageBridge.shared.setLowMemoryMode(newValue)
                                }

                            // Status indicator
                            if attentionSliceSize > 0 || lowMemoryMode {
                                HStack {
                                    Image(systemName: "checkmark.circle.fill")
                                        .foregroundColor(.orange)
                                    Text("Memory optimization active")
                                        .font(.caption)
                                        .foregroundColor(.orange)
                                }
                            }

                            Divider()

                            // Memory usage estimate
                            HStack {
                                Text("Estimated VRAM:")
                                    .font(.caption)
                                Text(estimatedMemoryUsage)
                                    .font(.caption)
                                    .foregroundColor(.orange)
                                    .bold()
                            }
                        }
                        .padding(8)
                    }

                    Spacer()

                    // Generate button
                    Button(action: generateImage) {
                        HStack {
                            if isGenerating {
                                ProgressView()
                                    .scaleEffect(0.7)
                            }
                            Text(isGenerating ? "Generating..." : "Generate Image")
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!modelsLoaded || isGenerating || prompt.isEmpty)

                    // Status
                    Text(statusMessage)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding()
            }
            .frame(minWidth: 320, maxWidth: 420)

            // Right panel - Image preview
            VStack {
                if let image = generatedImage {
                    Image(nsImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .background(Color(NSColor.windowBackgroundColor))

                    HStack {
                        Button("Save Image") {
                            saveImage(image)
                        }
                        Button("Copy to Clipboard") {
                            copyImageToClipboard(image)
                        }
                    }
                    .padding(.bottom)
                } else {
                    Text("Generated image will appear here")
                        .foregroundColor(.secondary)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
            .frame(minWidth: 400)
        }
    }

    private var attentionSliceDescription: String {
        switch Int(attentionSliceSize) {
        case 0: return "No slicing - fastest, uses most memory"
        case 1: return "Process 1 head at a time - slowest, minimum memory"
        case 2...4: return "Low memory mode - slower but uses ~12-16GB"
        case 5...8: return "Medium memory - balanced speed/memory"
        default: return "High slicing - more memory savings"
        }
    }

    private var estimatedMemoryUsage: String {
        let baseMemory: Double
        switch Int(width) {
        case 256: baseMemory = 12
        case 512: baseMemory = 20
        case 768: baseMemory = 28
        default: baseMemory = 40
        }

        var adjusted = baseMemory
        if attentionSliceSize >= 4 {
            adjusted *= 0.75
        }
        if lowMemoryMode {
            adjusted -= 7.5
        }

        return String(format: "~%.0f GB for %dx%d", max(adjusted, 6), Int(width), Int(height))
    }

    private func applyPreset(_ preset: VRAMPreset) {
        preset.apply()
        attentionSliceSize = Double(preset.attentionSliceSize)
        lowMemoryMode = preset.lowMemoryMode
    }

    private func loadModels() {
        isLoading = true
        statusMessage = "Loading models..."

        DispatchQueue.global(qos: .userInitiated).async {
            let success = ZImageBridge.shared.loadImageModels(modelDir: modelDirectory)

            DispatchQueue.main.async {
                isLoading = false
                modelsLoaded = success
                statusMessage = success ? "Models loaded successfully" : "Failed to load models: \(ZImageBridge.shared.lastError ?? "Unknown error")"
            }
        }
    }

    private func generateImage() {
        isGenerating = true
        statusMessage = "Generating image..."

        // Apply settings
        ZImageBridge.shared.setNumSteps(Int32(numSteps))
        ZImageBridge.shared.setAttentionSliceSize(Int32(attentionSliceSize))
        ZImageBridge.shared.setLowMemoryMode(lowMemoryMode)

        if useSeed, let seedValue = UInt64(seed) {
            ZImageBridge.shared.setSeed(seedValue)
        } else {
            ZImageBridge.shared.setSeed(0)
        }

        let tempPath = FileManager.default.temporaryDirectory
            .appendingPathComponent("z_image_\(UUID().uuidString).png")
            .path

        DispatchQueue.global(qos: .userInitiated).async {
            let success = ZImageBridge.shared.generateImage(
                prompt: prompt,
                width: Int32(width),
                height: Int32(height),
                modelDir: modelDirectory,
                outputPath: tempPath
            )

            DispatchQueue.main.async {
                isGenerating = false

                if success {
                    if let image = NSImage(contentsOfFile: tempPath) {
                        generatedImage = image
                        statusMessage = "Image generated successfully"
                    } else {
                        statusMessage = "Failed to load generated image"
                    }
                } else {
                    statusMessage = "Generation failed: \(ZImageBridge.shared.lastError ?? "Unknown error")"
                }
            }
        }
    }

    private func saveImage(_ image: NSImage) {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [UTType.png]
        panel.nameFieldStringValue = "generated_image.png"

        if panel.runModal() == .OK, let url = panel.url {
            if let tiffData = image.tiffRepresentation,
               let bitmap = NSBitmapImageRep(data: tiffData),
               let pngData = bitmap.representation(using: .png, properties: [:]) {
                try? pngData.write(to: url)
            }
        }
    }

    private func copyImageToClipboard(_ image: NSImage) {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.writeObjects([image])
    }
}

// MARK: - Chat View

struct ChatView: View {
    @Binding var modelDirectory: String

    @State private var chatHistory: [(role: String, content: String)] = []
    @State private var inputText = ""
    @State private var isChatting = false
    @State private var chatModelLoaded = false
    @State private var maxTokens: Double = 512
    @State private var temperature: Double = 0.7

    var body: some View {
        VStack {
            // Chat model status
            HStack {
                Circle()
                    .fill(chatModelLoaded ? Color.green : Color.red)
                    .frame(width: 10, height: 10)
                Text(chatModelLoaded ? "Chat Model Loaded" : "Chat Model Not Loaded")
                    .font(.caption)
                Spacer()
                if !chatModelLoaded && !modelDirectory.isEmpty {
                    Button("Load Chat Model") {
                        loadChatModel()
                    }
                }
            }
            .padding()

            // Settings row
            HStack {
                Text("Max Tokens:")
                    .font(.caption)
                Slider(value: $maxTokens, in: 32...1024, step: 32)
                Text("\(Int(maxTokens))")
                    .font(.caption)
                    .frame(width: 40)

                Divider()
                    .frame(height: 20)

                Text("Temperature:")
                    .font(.caption)
                Slider(value: $temperature, in: 0...1.5, step: 0.1)
                Text(String(format: "%.1f", temperature))
                    .font(.caption)
                    .frame(width: 30)
            }
            .padding(.horizontal)

            // Chat messages
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 12) {
                    ForEach(Array(chatHistory.enumerated()), id: \.offset) { _, message in
                        ChatBubble(role: message.role, content: message.content)
                    }
                }
                .padding()
            }

            // Input
            HStack {
                TextField("Type a message...", text: $inputText)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit {
                        sendMessage()
                    }

                Button(action: sendMessage) {
                    Image(systemName: "paperplane.fill")
                }
                .disabled(!chatModelLoaded || isChatting || inputText.isEmpty)
            }
            .padding()
        }
    }

    private func loadChatModel() {
        DispatchQueue.global(qos: .userInitiated).async {
            let success = ZImageBridge.shared.loadChatModel(modelDir: modelDirectory)
            DispatchQueue.main.async {
                chatModelLoaded = success
            }
        }
    }

    private func sendMessage() {
        guard !inputText.isEmpty else { return }

        let userMessage = inputText
        chatHistory.append((role: "user", content: userMessage))
        inputText = ""
        isChatting = true

        DispatchQueue.global(qos: .userInitiated).async {
            let response = ZImageBridge.shared.chat(
                prompt: userMessage,
                maxTokens: Int32(maxTokens),
                temperature: Float(temperature)
            )

            DispatchQueue.main.async {
                isChatting = false
                if let response = response {
                    chatHistory.append((role: "assistant", content: response))
                } else {
                    chatHistory.append((role: "assistant", content: "Error: Failed to generate response"))
                }
            }
        }
    }
}

struct ChatBubble: View {
    let role: String
    let content: String

    var body: some View {
        HStack {
            if role == "user" { Spacer() }

            Text(content)
                .padding(12)
                .background(role == "user" ? Color.accentColor : Color(NSColor.controlBackgroundColor))
                .foregroundColor(role == "user" ? .white : .primary)
                .cornerRadius(16)

            if role == "assistant" { Spacer() }
        }
    }
}

// MARK: - Settings View

struct SettingsView: View {
    @Binding var modelDirectory: String
    @Binding var modelsLoaded: Bool
    @Binding var isLoading: Bool
    @Binding var statusMessage: String

    @State private var attentionSliceSize: Int32 = 0
    @State private var lowMemoryMode = false
    @State private var numSteps: Int32 = 8

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                // Model Directory
                GroupBox("Model Directory") {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            TextField("Path to models", text: $modelDirectory)
                                .textFieldStyle(.roundedBorder)

                            Button("Browse...") {
                                selectModelDirectory()
                            }
                        }

                        Text("Directory should contain: ae.bpk, qwen3_4b_text_encoder.bpk, z_image_turbo_bf16.bpk")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .padding()
                }

                // Model Status
                GroupBox("Model Status") {
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Text("Image Models:")
                            Spacer()
                            if modelsLoaded {
                                Text("Loaded")
                                    .foregroundColor(.green)
                                Button("Unload") {
                                    unloadModels()
                                }
                            } else {
                                Text("Not Loaded")
                                    .foregroundColor(.red)
                                if !modelDirectory.isEmpty {
                                    Button("Load") {
                                        loadModels()
                                    }
                                    .disabled(isLoading)
                                }
                            }
                        }

                        if isLoading {
                            ProgressView()
                        }
                    }
                    .padding()
                }

                // Memory Optimization
                GroupBox("Memory Optimization") {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Adjust these settings if you run out of VRAM")
                            .font(.caption)
                            .foregroundColor(.orange)

                        // VRAM Presets
                        VStack(alignment: .leading, spacing: 4) {
                            Text("VRAM Presets")
                                .font(.subheadline)
                            HStack {
                                ForEach(VRAMPreset.allCases, id: \.self) { preset in
                                    Button(preset.rawValue) {
                                        preset.apply()
                                        attentionSliceSize = preset.attentionSliceSize
                                        lowMemoryMode = preset.lowMemoryMode
                                    }
                                    .buttonStyle(.bordered)
                                }
                            }
                        }

                        Divider()

                        // Manual controls
                        HStack {
                            Text("Attention Slice Size:")
                            Picker("", selection: $attentionSliceSize) {
                                Text("0 (No slicing)").tag(Int32(0))
                                Text("2").tag(Int32(2))
                                Text("4").tag(Int32(4))
                                Text("8").tag(Int32(8))
                                Text("16").tag(Int32(16))
                            }
                            .onChange(of: attentionSliceSize) { _, newValue in
                                ZImageBridge.shared.setAttentionSliceSize(newValue)
                            }
                        }

                        Toggle("Low Memory Mode (unloads text encoder during diffusion)", isOn: $lowMemoryMode)
                            .toggleStyle(.checkbox)
                            .onChange(of: lowMemoryMode) { _, newValue in
                                ZImageBridge.shared.setLowMemoryMode(newValue)
                            }

                        // Memory guide
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Memory Usage Guide")
                                .font(.subheadline)
                            Grid(alignment: .leading) {
                                GridRow {
                                    Text("Resolution").bold()
                                    Text("No Optimization").bold()
                                    Text("Slice=4").bold()
                                    Text("+ Low Mem").bold()
                                }
                                GridRow {
                                    Text("256x256")
                                    Text("~12 GB")
                                    Text("~10 GB")
                                    Text("~6 GB")
                                }
                                GridRow {
                                    Text("512x512")
                                    Text("~20 GB")
                                    Text("~16 GB")
                                    Text("~10 GB")
                                }
                                GridRow {
                                    Text("768x768")
                                    Text("~28 GB")
                                    Text("~22 GB")
                                    Text("~14 GB")
                                }
                                GridRow {
                                    Text("1024x1024")
                                    Text("~40 GB")
                                    Text("~30 GB")
                                    Text("~18 GB")
                                }
                            }
                            .font(.caption)
                        }
                    }
                    .padding()
                }

                // Default Generation Settings
                GroupBox("Default Generation Settings") {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("Inference Steps:")
                            Picker("", selection: $numSteps) {
                                Text("4").tag(Int32(4))
                                Text("6").tag(Int32(6))
                                Text("8 (default)").tag(Int32(8))
                                Text("12").tag(Int32(12))
                                Text("16").tag(Int32(16))
                                Text("20").tag(Int32(20))
                            }
                            .onChange(of: numSteps) { _, newValue in
                                ZImageBridge.shared.setNumSteps(newValue)
                            }
                        }
                        Text("Z-Image Turbo is optimized for 4-8 steps.")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .padding()
                }
            }
            .padding()
        }
    }

    private func selectModelDirectory() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false

        if panel.runModal() == .OK, let url = panel.url {
            modelDirectory = url.path
        }
    }

    private func loadModels() {
        isLoading = true
        statusMessage = "Loading models..."

        DispatchQueue.global(qos: .userInitiated).async {
            let success = ZImageBridge.shared.loadImageModels(modelDir: modelDirectory)

            DispatchQueue.main.async {
                isLoading = false
                modelsLoaded = success
                statusMessage = success ? "Models loaded" : "Failed to load models"
            }
        }
    }

    private func unloadModels() {
        _ = ZImageBridge.shared.unloadImageModels()
        modelsLoaded = false
        statusMessage = "Models unloaded"
    }
}

#Preview {
    ContentView()
}
