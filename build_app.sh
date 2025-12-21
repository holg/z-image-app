#!/bin/bash
set -e

# Build the Z-Image macOS App
# This script builds the Rust library and Swift app

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Building Z-Image macOS App ===${NC}"

# Step 1: Build the Rust library
echo -e "${YELLOW}Building Rust library...${NC}"
cargo build --release --lib

# Check if library was built
DYLIB_PATH="target/release/libz_image_app.dylib"
if [ ! -f "$DYLIB_PATH" ]; then
    echo -e "${RED}Error: Rust library not found at $DYLIB_PATH${NC}"
    exit 1
fi

echo -e "${GREEN}Rust library built successfully${NC}"

# Step 2: Create the app bundle directory structure
APP_NAME="ZImage"
APP_BUNDLE="$SCRIPT_DIR/$APP_NAME.app"
CONTENTS_DIR="$APP_BUNDLE/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"
RESOURCES_DIR="$CONTENTS_DIR/Resources"
FRAMEWORKS_DIR="$CONTENTS_DIR/Frameworks"

echo -e "${YELLOW}Creating app bundle structure...${NC}"
rm -rf "$APP_BUNDLE"
mkdir -p "$MACOS_DIR"
mkdir -p "$RESOURCES_DIR"
mkdir -p "$FRAMEWORKS_DIR"

# Step 3: Copy the dylib
echo -e "${YELLOW}Copying library...${NC}"
cp "$DYLIB_PATH" "$FRAMEWORKS_DIR/libz_image_app.dylib"

# Step 4: Create Info.plist
echo -e "${YELLOW}Creating Info.plist...${NC}"
cat > "$CONTENTS_DIR/Info.plist" << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>ZImage</string>
    <key>CFBundleIdentifier</key>
    <string>com.zimage.app</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>Z-Image</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>LSMinimumSystemVersion</key>
    <string>13.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSSupportsAutomaticGraphicsSwitching</key>
    <true/>
</dict>
</plist>
EOF

# Step 5: Build Swift app
echo -e "${YELLOW}Building Swift app...${NC}"

# Create a temporary Swift package for building
SWIFT_BUILD_DIR="$SCRIPT_DIR/.swift-build"
rm -rf "$SWIFT_BUILD_DIR"
mkdir -p "$SWIFT_BUILD_DIR/Sources/ZImage"

# Copy Swift sources
cp ZImageApp/*.swift "$SWIFT_BUILD_DIR/Sources/ZImage/"

# Create Package.swift
cat > "$SWIFT_BUILD_DIR/Package.swift" << EOF
// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "ZImage",
    platforms: [.macOS(.v13)],
    products: [
        .executable(name: "ZImage", targets: ["ZImage"])
    ],
    targets: [
        .executableTarget(
            name: "ZImage",
            path: "Sources/ZImage",
            swiftSettings: [
                .unsafeFlags([
                    "-I", "$SCRIPT_DIR",
                    "-L", "$SCRIPT_DIR/target/release",
                    "-lz_image_app"
                ])
            ],
            linkerSettings: [
                .unsafeFlags([
                    "-L", "$SCRIPT_DIR/target/release",
                    "-lz_image_app",
                    "-Xlinker", "-rpath", "-Xlinker", "@executable_path/../Frameworks"
                ])
            ]
        )
    ]
)
EOF

# Build with Swift
cd "$SWIFT_BUILD_DIR"
swift build -c release

# Copy executable
cp ".build/release/ZImage" "$MACOS_DIR/"
cd "$SCRIPT_DIR"

# Step 6: Fix library paths
echo -e "${YELLOW}Fixing library paths...${NC}"
install_name_tool -change \
    "target/release/libz_image_app.dylib" \
    "@executable_path/../Frameworks/libz_image_app.dylib" \
    "$MACOS_DIR/ZImage" 2>/dev/null || true

# Clean up
rm -rf "$SWIFT_BUILD_DIR"

echo -e "${GREEN}=== Build Complete ===${NC}"
echo -e "App bundle created at: ${YELLOW}$APP_BUNDLE${NC}"
echo ""
echo "To run the app:"
echo "  open $APP_BUNDLE"
echo ""
echo "Or run the egui version instead:"
echo "  cargo run --release --bin gui --features egui"
