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
        .frame(minWidth: 800, minHeight: 600)
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
    @State private var generatedImage: NSImage?
    @State private var isGenerating = false

    var body: some View {
        HSplitView {
            // Left panel - Controls
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
                        .frame(height: 100)
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
                }

                // Generation settings
                VStack(alignment: .leading, spacing: 8) {
                    Text("Settings")
                        .font(.headline)

                    HStack {
                        Text("Steps:")
                        Slider(value: $numSteps, in: 4...20, step: 1)
                        Text("\(Int(numSteps))")
                            .frame(width: 30)
                    }

                    HStack {
                        Text("Seed:")
                        TextField("0 for random", text: $seed)
                            .textFieldStyle(.roundedBorder)
                            .frame(width: 120)
                    }
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
            .frame(minWidth: 300, maxWidth: 400)

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

        let seedValue = Int64(seed) ?? 0
        let settings = ZImageGenerationSettings.custom(
            numInferenceSteps: Int32(numSteps),
            seed: seedValue
        )

        let tempPath = FileManager.default.temporaryDirectory
            .appendingPathComponent("z_image_\(UUID().uuidString).png")
            .path

        DispatchQueue.global(qos: .userInitiated).async {
            let success = ZImageBridge.shared.generateImageCached(
                prompt: prompt,
                width: Int32(width),
                height: Int32(height),
                outputPath: tempPath,
                settings: settings
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
            let response = ZImageBridge.shared.chat(prompt: userMessage, maxTokens: 512)

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

    var body: some View {
        Form {
            Section("Model Directory") {
                HStack {
                    TextField("Path to models", text: $modelDirectory)
                        .textFieldStyle(.roundedBorder)

                    Button("Browse...") {
                        selectModelDirectory()
                    }
                }

                Text("Directory should contain: ae.bpk, qwen3_4b_text_encoder.bpk, z_image_turbo.bpk")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Section("Model Status") {
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

            Section("About") {
                HStack {
                    Text("Library Version:")
                    Spacer()
                    Text(ZImageBridge.shared.version)
                }
            }
        }
        .padding()
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
