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
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default backend
BACKEND="${1:-candle-metal}"

# Show usage
show_usage() {
    echo "Usage: $0 [BACKEND]"
    echo ""
    echo "Available backends:"
    echo "  candle-metal   - Candle with Metal (default, stable)"
    echo "  wgpu-metal     - WGPU with Metal (has flash attention, experimental)"
    echo "  cpu            - CPU only (ndarray, slow but universal)"
    echo ""
    echo "Examples:"
    echo "  $0                  # Build with candle-metal (default)"
    echo "  $0 wgpu-metal       # Build with WGPU Metal + flash attention"
    echo "  $0 cpu              # Build with CPU backend"
}

# Parse arguments
case "$BACKEND" in
    candle-metal|metal)
        BACKEND="candle-metal"
        CARGO_FEATURES="--features metal"
        BACKEND_DESC="Candle + Metal (naive attention)"
        ;;
    wgpu-metal|wgpu)
        BACKEND="wgpu-metal"
        CARGO_FEATURES="--no-default-features --features wgpu-metal"
        BACKEND_DESC="WGPU + Metal (flash attention)"
        ;;
    cpu|ndarray)
        BACKEND="cpu"
        CARGO_FEATURES="--no-default-features --features cpu"
        BACKEND_DESC="CPU only (ndarray)"
        ;;
    -h|--help|help)
        show_usage
        exit 0
        ;;
    *)
        echo -e "${RED}Unknown backend: $BACKEND${NC}"
        show_usage
        exit 1
        ;;
esac

echo -e "${GREEN}=== Building Z-Image macOS App ===${NC}"
echo -e "${CYAN}Backend: $BACKEND_DESC${NC}"
echo ""

# Step 1: Build the Rust library
echo -e "${YELLOW}Building Rust library...${NC}"
cargo build --release --lib $CARGO_FEATURES

# Check if library was built
DYLIB_PATH="target/release/libz_image_ffi.dylib"
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
cp "$DYLIB_PATH" "$FRAMEWORKS_DIR/libz_image_ffi.dylib"

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

# Step 5: Build Swift app using swiftc directly
echo -e "${YELLOW}Building Swift app...${NC}"

# Create a bridging header that imports the FFI header directly
BRIDGE_HEADER="/tmp/ZImage-Bridging-Header.h"
cat > "$BRIDGE_HEADER" << EOF
#ifndef ZImage_Bridging_Header_h
#define ZImage_Bridging_Header_h

#include "$SCRIPT_DIR/z_image_ffi.h"

#endif
EOF

# Compile Swift sources (exclude module.modulemap)
SWIFT_FILES=$(find "$SCRIPT_DIR/ZImageApp" -name "*.swift" -type f)

swiftc \
    -O \
    -whole-module-optimization \
    -target arm64-apple-macos13.0 \
    -sdk $(xcrun --show-sdk-path) \
    -import-objc-header "$BRIDGE_HEADER" \
    -I "$SCRIPT_DIR" \
    -L "$SCRIPT_DIR/target/release" \
    -lz_image_ffi \
    -Xlinker -rpath -Xlinker "@executable_path/../Frameworks" \
    -o "$MACOS_DIR/ZImage" \
    $SWIFT_FILES

# Clean up bridging header
rm -f "$BRIDGE_HEADER"

echo -e "${GREEN}Swift app compiled successfully${NC}"

# Step 6: Fix library paths
echo -e "${YELLOW}Fixing library paths...${NC}"
install_name_tool -change \
    "target/release/libz_image_ffi.dylib" \
    "@executable_path/../Frameworks/libz_image_ffi.dylib" \
    "$MACOS_DIR/ZImage" 2>/dev/null || true

# Also fix the library's own install name
install_name_tool -id \
    "@executable_path/../Frameworks/libz_image_ffi.dylib" \
    "$FRAMEWORKS_DIR/libz_image_ffi.dylib" 2>/dev/null || true

# Clean up
rm -f "$BRIDGE_HEADER"

echo -e "${GREEN}=== Build Complete ===${NC}"
echo -e "App bundle created at: ${YELLOW}$APP_BUNDLE${NC}"
echo ""
echo "To run the app:"
echo "  open $APP_BUNDLE"
echo ""
echo "Or run the egui version instead:"
echo "  cargo run --release --bin gui --features egui"
