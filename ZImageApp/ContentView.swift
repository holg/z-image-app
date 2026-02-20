import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @State private var selectedTab = 0
    @AppStorage("modelDirectory") private var modelDirectory: String = "/Volumes/tb3ssd/volume/burn_models"
    @State private var modelsLoaded = false
    @State private var isLoading = false
    @State private var statusMessage = "Select model directory to begin"
    @StateObject private var logManager = LogManager.shared
    @StateObject private var historyManager = HistoryManager.shared

    // Persistent state for Image tab (so it survives tab switches)
    @State private var imagePrompt: String = ""
    @State private var imageWidth: Double = 512
    @State private var imageHeight: Double = 512
    @State private var imageNumSteps: Double = 8
    @State private var imageSeed: String = "0"
    @State private var imageUseSeed: Bool = false
    @State private var generatedImage: NSImage? = nil

    var body: some View {
        VStack(spacing: 0) {
            // Tab bar
            HStack(spacing: 0) {
                TabButton(title: "Image", icon: "photo", isSelected: selectedTab == 0) {
                    selectedTab = 0
                }
                TabButton(title: "Video", icon: "film", isSelected: selectedTab == 1) {
                    selectedTab = 1
                }
                TabButton(title: "History", icon: "clock.arrow.circlepath", isSelected: selectedTab == 2) {
                    selectedTab = 2
                }
                TabButton(title: "Chat", icon: "bubble.left", isSelected: selectedTab == 3) {
                    selectedTab = 3
                }
                TabButton(title: "Settings", icon: "gear", isSelected: selectedTab == 4) {
                    selectedTab = 4
                }
                TabButton(title: "Logs", icon: "terminal", isSelected: selectedTab == 5) {
                    selectedTab = 5
                }
            }
            .padding(.horizontal)
            .padding(.top, 8)

            Divider()
                .padding(.top, 8)

            // Content - using ZStack to keep views alive
            ZStack {
                ImageGenerationView(
                    modelDirectory: $modelDirectory,
                    modelsLoaded: $modelsLoaded,
                    isLoading: $isLoading,
                    statusMessage: $statusMessage,
                    logManager: logManager,
                    historyManager: historyManager,
                    prompt: $imagePrompt,
                    width: $imageWidth,
                    height: $imageHeight,
                    numSteps: $imageNumSteps,
                    seed: $imageSeed,
                    useSeed: $imageUseSeed,
                    generatedImage: $generatedImage
                )
                .opacity(selectedTab == 0 ? 1 : 0)
                .allowsHitTesting(selectedTab == 0)

                VideoGenerationView(
                    modelDirectory: $modelDirectory,
                    imageModelsLoaded: $modelsLoaded,
                    logManager: logManager
                )
                .opacity(selectedTab == 1 ? 1 : 0)
                .allowsHitTesting(selectedTab == 1)

                HistoryView(
                    historyManager: historyManager,
                    selectedTab: $selectedTab,
                    imagePrompt: $imagePrompt,
                    imageWidth: $imageWidth,
                    imageHeight: $imageHeight,
                    imageNumSteps: $imageNumSteps
                )
                .opacity(selectedTab == 2 ? 1 : 0)
                .allowsHitTesting(selectedTab == 2)

                ChatView(modelDirectory: $modelDirectory)
                    .opacity(selectedTab == 3 ? 1 : 0)
                    .allowsHitTesting(selectedTab == 3)

                SettingsView(
                    modelDirectory: $modelDirectory,
                    modelsLoaded: $modelsLoaded,
                    isLoading: $isLoading,
                    statusMessage: $statusMessage,
                    logManager: logManager
                )
                .opacity(selectedTab == 4 ? 1 : 0)
                .allowsHitTesting(selectedTab == 4)

                LogView(logManager: logManager)
                    .opacity(selectedTab == 5 ? 1 : 0)
                    .allowsHitTesting(selectedTab == 5)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .frame(minWidth: 900, minHeight: 700)
        .onAppear {
            logManager.log("Z-Image App started")
            logManager.log("Initialize GPU device...")
            z_image_init()
            logManager.log("GPU device initialized")
            historyManager.loadHistory()
            logManager.log("Loaded \(historyManager.items.count) history items")
        }
    }
}

// MARK: - Log Manager

class LogManager: ObservableObject {
    static let shared = LogManager()

    @Published var logs: [LogEntry] = []

    struct LogEntry: Identifiable {
        let id = UUID()
        let timestamp: Date
        let message: String
    }

    func log(_ message: String) {
        DispatchQueue.main.async {
            let entry = LogEntry(timestamp: Date(), message: message)
            self.logs.append(entry)
            // Keep only last 500 entries
            if self.logs.count > 500 {
                self.logs.removeFirst(self.logs.count - 500)
            }
        }
    }

    func clear() {
        DispatchQueue.main.async {
            self.logs.removeAll()
        }
    }
}

// MARK: - History Manager

struct HistoryItem: Identifiable, Codable, Hashable {
    let id: String
    let prompt: String
    let width: Int
    let height: Int
    let steps: Int?  // Optional for backwards compatibility with egui history
    let timestamp: Double  // Seconds since reference date (for egui compatibility)
    let imagePath: String
    let generationTime: Double

    var date: Date {
        Date(timeIntervalSinceReferenceDate: timestamp)
    }

    init(prompt: String, width: Int, height: Int, steps: Int, imagePath: String, generationTime: Double) {
        self.id = UUID().uuidString
        self.prompt = prompt
        self.width = width
        self.height = height
        self.steps = steps
        self.timestamp = Date().timeIntervalSinceReferenceDate
        self.imagePath = imagePath
        self.generationTime = generationTime
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }

    static func == (lhs: HistoryItem, rhs: HistoryItem) -> Bool {
        lhs.id == rhs.id
    }
}

class HistoryManager: ObservableObject {
    static let shared = HistoryManager()

    @Published var items: [HistoryItem] = []

    private var appDir: URL {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let dir = appSupport.appendingPathComponent("ZImageApp", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private var historyDir: URL {
        let dir = appDir.appendingPathComponent("history", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private var historyFile: URL {
        appDir.appendingPathComponent("history.json")
    }

    func loadHistory() {
        DispatchQueue.global(qos: .background).async {
            var loadedItems: [HistoryItem] = []

            // Load from single history.json file (egui format)
            if let data = try? Data(contentsOf: self.historyFile),
               let items = try? JSONDecoder().decode([HistoryItem].self, from: data) {
                for item in items {
                    // Check if image still exists
                    if FileManager.default.fileExists(atPath: item.imagePath) {
                        loadedItems.append(item)
                    }
                }
            }

            // Sort by timestamp, newest first
            loadedItems.sort { $0.timestamp > $1.timestamp }

            DispatchQueue.main.async {
                self.items = loadedItems
            }
        }
    }

    private func saveHistory() {
        DispatchQueue.global(qos: .background).async {
            if let data = try? JSONEncoder().encode(self.items) {
                try? data.write(to: self.historyFile)
            }
        }
    }

    func addItem(prompt: String, width: Int, height: Int, steps: Int, tempImagePath: String, generationTime: Double) {
        let id = UUID().uuidString
        let imagePath = historyDir.appendingPathComponent("\(id).png")

        // Copy image to history
        do {
            try FileManager.default.copyItem(atPath: tempImagePath, toPath: imagePath.path)
        } catch {
            print("Failed to copy image to history: \(error)")
            return
        }

        let item = HistoryItem(
            prompt: prompt,
            width: width,
            height: height,
            steps: steps,
            imagePath: imagePath.path,
            generationTime: generationTime
        )

        DispatchQueue.main.async {
            self.items.insert(item, at: 0)
            self.saveHistory()
        }
    }

    func deleteItem(_ item: HistoryItem) {
        // Delete image file
        try? FileManager.default.removeItem(atPath: item.imagePath)

        DispatchQueue.main.async {
            self.items.removeAll { $0.id == item.id }
            self.saveHistory()
        }
    }
}

// MARK: - History View

struct HistoryView: View {
    @ObservedObject var historyManager: HistoryManager
    @Binding var selectedTab: Int
    @Binding var imagePrompt: String
    @Binding var imageWidth: Double
    @Binding var imageHeight: Double
    @Binding var imageNumSteps: Double

    @State private var selectedItem: HistoryItem?

    private let dateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateStyle = .short
        formatter.timeStyle = .medium
        return formatter
    }()

    var body: some View {
        HStack(spacing: 0) {
            // Left: List of history items
            VStack(alignment: .leading, spacing: 0) {
                HStack {
                    Image(systemName: "clock.arrow.circlepath")
                    Text("History")
                        .font(.headline)
                    Spacer()
                    Text("\(historyManager.items.count) images")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding()

                Divider()

                if historyManager.items.isEmpty {
                    VStack {
                        Spacer()
                        Text("No images generated yet")
                            .foregroundColor(.secondary)
                        Spacer()
                    }
                } else {
                    List(historyManager.items, selection: $selectedItem) { item in
                        VStack(alignment: .leading, spacing: 4) {
                            Text(item.prompt)
                                .lineLimit(2)
                                .font(.body)
                            HStack {
                                Text("\(item.width)x\(item.height)")
                                if let steps = item.steps {
                                    Text("•")
                                    Text("\(steps) steps")
                                }
                                Text("•")
                                Text(dateFormatter.string(from: item.date))
                            }
                            .font(.caption)
                            .foregroundColor(.secondary)
                        }
                        .padding(.vertical, 4)
                        .contentShape(Rectangle())
                        .onTapGesture {
                            selectedItem = item
                        }
                    }
                }
            }
            .frame(width: 300)
            .background(Color(NSColor.windowBackgroundColor))

            Divider()

            // Right: Selected item details
            if let item = selectedItem {
                VStack(spacing: 16) {
                    // Image preview
                    if let image = NSImage(contentsOfFile: item.imagePath) {
                        Image(nsImage: image)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(maxHeight: 400)
                            .cornerRadius(8)
                    }

                    // Details
                    GroupBox("Details") {
                        VStack(alignment: .leading, spacing: 8) {
                            LabeledContent("Prompt") {
                                Text(item.prompt)
                                    .textSelection(.enabled)
                            }
                            LabeledContent("Size") {
                                Text("\(item.width) x \(item.height)")
                            }
                            if let steps = item.steps {
                                LabeledContent("Steps") {
                                    Text("\(steps)")
                                }
                            }
                            LabeledContent("Generated") {
                                Text(dateFormatter.string(from: item.date))
                            }
                            LabeledContent("Time") {
                                Text(String(format: "%.1f seconds", item.generationTime))
                            }
                        }
                        .padding(8)
                    }

                    // Actions
                    HStack {
                        Button("Use This Prompt") {
                            imagePrompt = item.prompt
                            imageWidth = Double(item.width)
                            imageHeight = Double(item.height)
                            imageNumSteps = Double(item.steps ?? 8)
                            selectedTab = 0 // Switch to Image tab
                        }
                        .buttonStyle(.borderedProminent)

                        Button("Open in Finder") {
                            NSWorkspace.shared.selectFile(item.imagePath, inFileViewerRootedAtPath: "")
                        }

                        Button("Delete", role: .destructive) {
                            historyManager.deleteItem(item)
                            selectedItem = nil
                        }
                    }

                    Spacer()
                }
                .padding()
                .frame(maxWidth: .infinity)
            } else {
                VStack {
                    Spacer()
                    Text("Select an image from the list")
                        .foregroundColor(.secondary)
                    Spacer()
                }
                .frame(maxWidth: .infinity)
            }
        }
    }
}

// MARK: - Log View

struct LogView: View {
    @ObservedObject var logManager: LogManager
    @State private var autoScroll = true

    private let dateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss.SSS"
        return formatter
    }()

    var body: some View {
        VStack(spacing: 0) {
            // Toolbar
            HStack {
                Image(systemName: "terminal")
                    .font(.title2)
                Text("Logs")
                    .font(.headline)

                Spacer()

                Toggle("Auto-scroll", isOn: $autoScroll)
                    .toggleStyle(.checkbox)

                Button("Clear") {
                    logManager.clear()
                }

                Button("Copy All") {
                    let text = logManager.logs.map { "[\(dateFormatter.string(from: $0.timestamp))] \($0.message)" }.joined(separator: "\n")
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(text, forType: .string)
                }
            }
            .padding()
            .background(Color(NSColor.windowBackgroundColor))

            Divider()

            // Log content
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 2) {
                        ForEach(logManager.logs) { entry in
                            HStack(alignment: .top, spacing: 8) {
                                Text("[\(dateFormatter.string(from: entry.timestamp))]")
                                    .font(.system(.caption, design: .monospaced))
                                    .foregroundColor(.secondary)
                                Text(entry.message)
                                    .font(.system(.caption, design: .monospaced))
                                    .foregroundColor(colorForMessage(entry.message))
                                    .textSelection(.enabled)
                                Spacer()
                            }
                            .padding(.horizontal, 8)
                            .padding(.vertical, 1)
                            .id(entry.id)
                        }
                    }
                    .padding(.vertical, 8)
                }
                .background(Color(NSColor.textBackgroundColor))
                .onChange(of: logManager.logs.count) { _ in
                    if autoScroll, let lastId = logManager.logs.last?.id {
                        withAnimation {
                            proxy.scrollTo(lastId, anchor: .bottom)
                        }
                    }
                }
            }
        }
    }

    private func colorForMessage(_ message: String) -> Color {
        if message.contains("Error") || message.contains("error") || message.contains("failed") {
            return .red
        } else if message.contains("Warning") || message.contains("warning") {
            return .orange
        } else if message.contains("SUCCESS") || message.contains("complete") || message.contains("loaded") {
            return .green
        } else if message.starts(with: "[z-image]") || message.starts(with: "[qwen3]") {
            return .cyan
        }
        return .primary
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
    @ObservedObject var logManager: LogManager
    @ObservedObject var historyManager: HistoryManager

    // Persistent state (passed from parent)
    @Binding var prompt: String
    @Binding var width: Double
    @Binding var height: Double
    @Binding var numSteps: Double
    @Binding var seed: String
    @Binding var useSeed: Bool
    @Binding var generatedImage: NSImage?

    @State private var isGenerating = false

    // Memory settings
    @State private var selectedPreset: VRAMPreset = .fast
    @State private var attentionSliceSize: Double = 0
    @State private var lowMemoryMode = false

    var body: some View {
        HStack(spacing: 0) {
            // Left panel - Controls (fixed width)
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
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
                            .frame(height: 70)
                            .font(.body)
                            .padding(4)
                            .background(Color(NSColor.textBackgroundColor))
                            .cornerRadius(8)
                    }

                    // Size controls - more compact
                    GroupBox("Image Size") {
                        VStack(alignment: .leading, spacing: 6) {
                            HStack {
                                Text("W:")
                                    .frame(width: 20)
                                Slider(value: $width, in: 256...1024, step: 64)
                                Text("\(Int(width))")
                                    .frame(width: 40)
                                    .monospacedDigit()
                            }
                            HStack {
                                Text("H:")
                                    .frame(width: 20)
                                Slider(value: $height, in: 256...1024, step: 64)
                                Text("\(Int(height))")
                                    .frame(width: 40)
                                    .monospacedDigit()
                            }
                            HStack(spacing: 4) {
                                Button("256") { width = 256; height = 256 }
                                Button("512") { width = 512; height = 512 }
                                Button("768") { width = 768; height = 768 }
                                Button("1024") { width = 1024; height = 1024 }
                            }
                            .font(.caption)
                            .buttonStyle(.bordered)
                        }
                        .padding(4)
                    }

                    // Generation settings - compact
                    GroupBox("Generation") {
                        VStack(alignment: .leading, spacing: 6) {
                            HStack {
                                Text("Steps:")
                                Slider(value: $numSteps, in: 4...20, step: 1)
                                Text("\(Int(numSteps))")
                                    .frame(width: 25)
                                    .monospacedDigit()
                            }

                            HStack {
                                Toggle("Seed:", isOn: $useSeed)
                                    .toggleStyle(.checkbox)
                                TextField("", text: $seed)
                                    .textFieldStyle(.roundedBorder)
                                    .frame(width: 80)
                                    .disabled(!useSeed)
                                if useSeed {
                                    Button("Rand") {
                                        seed = String(UInt64.random(in: 1...UInt64.max))
                                    }
                                    .font(.caption)
                                }
                            }
                        }
                        .padding(4)
                    }

                    // Memory optimization - compact
                    GroupBox("Memory") {
                        VStack(alignment: .leading, spacing: 6) {
                            // VRAM Presets as buttons
                            HStack(spacing: 4) {
                                ForEach(VRAMPreset.allCases, id: \.self) { preset in
                                    Button(preset.shortName) {
                                        selectedPreset = preset
                                        applyPreset(preset)
                                    }
                                    .buttonStyle(.bordered)
                                    .tint(selectedPreset == preset ? .accentColor : .secondary)
                                }
                            }
                            .font(.caption)

                            HStack {
                                Text("Slice:")
                                Slider(value: $attentionSliceSize, in: 0...30, step: 1)
                                Text("\(Int(attentionSliceSize))")
                                    .frame(width: 25)
                                    .monospacedDigit()
                            }

                            Toggle("Low Mem Mode", isOn: $lowMemoryMode)
                                .toggleStyle(.checkbox)
                                .onChange(of: lowMemoryMode) { newValue in
                                    ZImageBridge.shared.setLowMemoryMode(newValue)
                                }

                            if attentionSliceSize > 0 || lowMemoryMode {
                                Text("~\(estimatedMemoryUsage)")
                                    .font(.caption)
                                    .foregroundColor(.orange)
                            }
                        }
                        .padding(4)
                    }

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
                        .padding(8)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!modelsLoaded || isGenerating || prompt.isEmpty)

                    // Status
                    Text(statusMessage)
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .lineLimit(2)
                }
                .padding(12)
            }
            .frame(width: 340)
            .background(Color(NSColor.windowBackgroundColor))

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
        // Base: models need ~20GB loaded (transformer 12GB + text encoder 8GB)
        // Low memory mode unloads text encoder during diffusion, reducing to ~12GB
        // Attention slicing reduces peak memory spikes during attention computation

        let modelMemory: Double = lowMemoryMode ? 12.0 : 20.0

        // Peak memory during generation depends on resolution
        let peakExtra: Double
        switch Int(width) {
        case 256: peakExtra = lowMemoryMode ? 2.0 : 4.0
        case 512: peakExtra = lowMemoryMode ? 4.0 : 8.0
        case 768: peakExtra = lowMemoryMode ? 6.0 : 12.0
        default: peakExtra = lowMemoryMode ? 8.0 : 16.0
        }

        // Attention slicing reduces peak extra
        var adjustedPeak = peakExtra
        if attentionSliceSize >= 4 {
            adjustedPeak *= 0.6
        } else if attentionSliceSize >= 2 {
            adjustedPeak *= 0.7
        }

        let total = modelMemory + adjustedPeak

        if lowMemoryMode {
            return String(format: "Peak ~%.0fGB (models unload to %.0fGB)", total, modelMemory)
        } else {
            return String(format: "Peak ~%.0fGB", total)
        }
    }

    private func applyPreset(_ preset: VRAMPreset) {
        preset.apply()
        attentionSliceSize = Double(preset.attentionSliceSize)
        lowMemoryMode = preset.lowMemoryMode
    }

    private func loadModels() {
        isLoading = true
        statusMessage = "Loading models..."
        logManager.log("[z-image] Loading models from \(modelDirectory)...")

        DispatchQueue.global(qos: .userInitiated).async {
            let errorMessage = ZImageBridge.shared.loadImageModels(modelDir: modelDirectory)

            DispatchQueue.main.async {
                isLoading = false
                if let error = errorMessage {
                    modelsLoaded = false
                    statusMessage = "Failed: \(error)"
                    logManager.log("[z-image] Error: \(error)")
                } else {
                    modelsLoaded = true
                    statusMessage = "Models loaded successfully"
                    logManager.log("[z-image] Models loaded successfully")
                }
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

        logManager.log("[z-image] Starting generation...")
        logManager.log("[z-image] Prompt: \"\(prompt)\"")
        logManager.log("[z-image] Size: \(Int(width))x\(Int(height)), Steps: \(Int(numSteps))")
        logManager.log("[z-image] Memory: slice_size=\(Int(attentionSliceSize)), low_memory=\(lowMemoryMode)")

        if useSeed, let seedValue = UInt64(seed) {
            ZImageBridge.shared.setSeed(seedValue)
            logManager.log("[z-image] Using fixed seed: \(seedValue)")
        } else {
            ZImageBridge.shared.setSeed(0)
            logManager.log("[z-image] Using random seed")
        }

        let tempPath = FileManager.default.temporaryDirectory
            .appendingPathComponent("z_image_\(UUID().uuidString).png")
            .path

        let startTime = Date()

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
                let elapsed = Date().timeIntervalSince(startTime)

                if success {
                    if let image = NSImage(contentsOfFile: tempPath) {
                        generatedImage = image
                        statusMessage = "Image generated successfully"
                        logManager.log("[z-image] Generation complete in \(String(format: "%.1f", elapsed))s")
                        logManager.log("[z-image] Saved to: \(tempPath)")

                        // Save to history
                        historyManager.addItem(
                            prompt: prompt,
                            width: Int(width),
                            height: Int(height),
                            steps: Int(numSteps),
                            tempImagePath: tempPath,
                            generationTime: elapsed
                        )
                        logManager.log("[z-image] Added to history")
                    } else {
                        statusMessage = "Failed to load generated image"
                        logManager.log("[z-image] Error: Failed to load generated image from \(tempPath)")
                    }
                } else {
                    let error = ZImageBridge.shared.lastError ?? "Unknown error"
                    statusMessage = "Generation failed: \(error)"
                    logManager.log("[z-image] Error: Generation failed: \(error)")
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
    @ObservedObject var logManager: LogManager

    @State private var attentionSliceSize: Int32 = 0
    @State private var lowMemoryMode = false
    @State private var numSteps: Int32 = 8
    @State private var isDownloading = false
    @State private var downloadProgress = ""

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

                        Text("Required: z_image_turbo_bf16.bpk, longcat_dit_bf16.bpk, wan_vae.bpk, umt5-xxl-enc-bf16.bpk")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .padding()
                }

                // Download Models from HuggingFace
                GroupBox("Download Models from HuggingFace") {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Download pre-converted models from holgt's HuggingFace repos")
                            .font(.caption)
                            .foregroundColor(.secondary)

                        if isDownloading {
                            HStack {
                                ProgressView()
                                    .scaleEffect(0.8)
                                Text(downloadProgress)
                                    .font(.caption)
                            }
                        }

                        // Model download buttons
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                VStack(alignment: .leading) {
                                    Text("ZImage Turbo (11GB)")
                                        .font(.subheadline)
                                    Text("holgt/z-image-turbo-burn")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                                Spacer()
                                Button("Download") {
                                    downloadModel(repo: "holgt/z-image-turbo-burn", file: "z_image_turbo_bf16.bpk")
                                }
                                .disabled(isDownloading || modelDirectory.isEmpty)
                            }

                            Divider()

                            HStack {
                                VStack(alignment: .leading) {
                                    Text("LongCat DiT (25GB)")
                                        .font(.subheadline)
                                    Text("holgt/longcat-dit-burn")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                                Spacer()
                                Button("Download") {
                                    downloadModel(repo: "holgt/longcat-dit-burn", file: "longcat_dit_bf16.bpk")
                                }
                                .disabled(isDownloading || modelDirectory.isEmpty)
                            }

                            Divider()

                            HStack {
                                VStack(alignment: .leading) {
                                    Text("WAN VAE (242MB)")
                                        .font(.subheadline)
                                    Text("holgt/wan-vae-burn")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                                Spacer()
                                Button("Download") {
                                    downloadModel(repo: "holgt/wan-vae-burn", file: "wan_vae.bpk")
                                }
                                .disabled(isDownloading || modelDirectory.isEmpty)
                            }

                            Divider()

                            HStack {
                                VStack(alignment: .leading) {
                                    Text("UMT5-XXL Encoder (11GB)")
                                        .font(.subheadline)
                                    Text("holgt/umt5-xxl-burn")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                                Spacer()
                                Button("Download") {
                                    downloadModel(repo: "holgt/umt5-xxl-burn", file: "umt5-xxl-enc-bf16.bpk")
                                }
                                .disabled(isDownloading || modelDirectory.isEmpty)
                            }

                            Divider()

                            HStack {
                                VStack(alignment: .leading) {
                                    Text("Qwen3 4B Text Encoder (8GB)")
                                        .font(.subheadline)
                                    Text("holgt/qwen3-4b-text-encoder-burn")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                                Spacer()
                                Button("Download") {
                                    downloadModel(repo: "holgt/qwen3-4b-text-encoder-burn", file: "qwen3_4b_text_encoder.bpk")
                                }
                                .disabled(isDownloading || modelDirectory.isEmpty)
                            }

                            Divider()

                            HStack {
                                VStack(alignment: .leading) {
                                    Text("Qwen3 0.6B Chat (1.4GB)")
                                        .font(.subheadline)
                                    Text("holgt/qwen3-0.6b-burn")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                                Spacer()
                                Button("Download") {
                                    downloadModel(repo: "holgt/qwen3-0.6b-burn", file: "qwen3_0.6b.bpk")
                                }
                                .disabled(isDownloading || modelDirectory.isEmpty)
                            }
                        }

                        Divider()

                        Button("Download All Models") {
                            downloadAllModels()
                        }
                        .disabled(isDownloading || modelDirectory.isEmpty)
                        .buttonStyle(.borderedProminent)

                        if modelDirectory.isEmpty {
                            Text("Set a model directory first before downloading")
                                .font(.caption)
                                .foregroundColor(.orange)
                        }
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
                            .onChange(of: attentionSliceSize) { newValue in
                                ZImageBridge.shared.setAttentionSliceSize(newValue)
                            }
                        }

                        Toggle("Low Memory Mode (unloads text encoder during diffusion)", isOn: $lowMemoryMode)
                            .toggleStyle(.checkbox)
                            .onChange(of: lowMemoryMode) { newValue in
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
                            .onChange(of: numSteps) { newValue in
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
        logManager.log("[z-image] Loading models from \(modelDirectory)...")

        DispatchQueue.global(qos: .userInitiated).async {
            let errorMessage = ZImageBridge.shared.loadImageModels(modelDir: modelDirectory)

            DispatchQueue.main.async {
                isLoading = false
                if let error = errorMessage {
                    modelsLoaded = false
                    statusMessage = "Failed: \(error)"
                    logManager.log("[z-image] Error: \(error)")
                } else {
                    modelsLoaded = true
                    statusMessage = "Models loaded"
                    logManager.log("[z-image] Models loaded successfully")
                }
            }
        }
    }

    private func unloadModels() {
        logManager.log("[z-image] Unloading models...")
        _ = ZImageBridge.shared.unloadImageModels()
        logManager.log("[z-image] Models unloaded")
        modelsLoaded = false
        statusMessage = "Models unloaded"
    }

    private func downloadModel(repo: String, file: String) {
        isDownloading = true
        downloadProgress = "Downloading \(file)..."
        logManager.log("[download] Starting download: \(repo)/\(file)")

        DispatchQueue.global(qos: .userInitiated).async {
            let outputPath = URL(fileURLWithPath: modelDirectory).appendingPathComponent(file).path

            // Use huggingface-cli to download
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/opt/homebrew/bin/huggingface-cli")
            process.arguments = ["download", repo, file, "--local-dir", modelDirectory]

            let pipe = Pipe()
            process.standardOutput = pipe
            process.standardError = pipe

            do {
                try process.run()
                process.waitUntilExit()

                DispatchQueue.main.async {
                    isDownloading = false
                    if process.terminationStatus == 0 {
                        downloadProgress = "Downloaded \(file)"
                        logManager.log("[download] Successfully downloaded \(file)")
                    } else {
                        let data = pipe.fileHandleForReading.readDataToEndOfFile()
                        let output = String(data: data, encoding: .utf8) ?? "Unknown error"
                        downloadProgress = "Failed to download \(file)"
                        logManager.log("[download] Error: \(output)")
                    }
                }
            } catch {
                DispatchQueue.main.async {
                    isDownloading = false
                    downloadProgress = "Error: \(error.localizedDescription)"
                    logManager.log("[download] Error: \(error.localizedDescription)")
                }
            }
        }
    }

    private func downloadAllModels() {
        isDownloading = true
        downloadProgress = "Downloading all models..."
        logManager.log("[download] Starting download of all models...")

        DispatchQueue.global(qos: .userInitiated).async {
            let models = [
                ("holgt/z-image-turbo-burn", "z_image_turbo_bf16.bpk"),
                ("holgt/longcat-dit-burn", "longcat_dit_bf16.bpk"),
                ("holgt/wan-vae-burn", "wan_vae.bpk"),
                ("holgt/umt5-xxl-burn", "umt5-xxl-enc-bf16.bpk"),
                ("holgt/qwen3-4b-text-encoder-burn", "qwen3_4b_text_encoder.bpk"),
                ("holgt/qwen3-0.6b-burn", "qwen3_0.6b.bpk")
            ]

            for (index, (repo, file)) in models.enumerated() {
                DispatchQueue.main.async {
                    downloadProgress = "Downloading \(file) (\(index + 1)/\(models.count))..."
                }

                let process = Process()
                process.executableURL = URL(fileURLWithPath: "/opt/homebrew/bin/huggingface-cli")
                process.arguments = ["download", repo, file, "--local-dir", modelDirectory]

                do {
                    try process.run()
                    process.waitUntilExit()

                    if process.terminationStatus == 0 {
                        DispatchQueue.main.async {
                            logManager.log("[download] Downloaded \(file)")
                        }
                    } else {
                        DispatchQueue.main.async {
                            logManager.log("[download] Failed to download \(file)")
                        }
                    }
                } catch {
                    DispatchQueue.main.async {
                        logManager.log("[download] Error downloading \(file): \(error.localizedDescription)")
                    }
                }
            }

            DispatchQueue.main.async {
                isDownloading = false
                downloadProgress = "All downloads complete"
                logManager.log("[download] All downloads complete")
            }
        }
    }
}

#Preview {
    ContentView()
}
