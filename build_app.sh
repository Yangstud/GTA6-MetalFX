#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$ROOT_DIR/build"
APP_BUNDLE="$ROOT_DIR/CM716FX.app"
CONTENTS="$APP_BUNDLE/Contents"
MACOS_DIR="$CONTENTS/MacOS"
RES_DIR="$CONTENTS/Resources"

SOURCES=(
  "$ROOT_DIR/Sources/CM716FX/Core/Config.swift"
  "$ROOT_DIR/Sources/CM716FX/Capture/CM716CaptureManager.swift"
  "$ROOT_DIR/Sources/CM716FX/Engine/VTFrameGenEngine.swift"
  "$ROOT_DIR/Sources/CM716FX/Engine/MetalPipelineEngine.swift"
  "$ROOT_DIR/Sources/CM716FX/UI/MainWindowController.swift"
  "$ROOT_DIR/Sources/CM716FX/AppDelegate.swift"
  "$ROOT_DIR/Sources/CM716FX/main.swift"
)

FRAMEWORKS=(
  -framework AppKit
  -framework AVFoundation
  -framework CoreMedia
  -framework CoreVideo
  -framework Metal
  -framework MetalFX
  -framework QuartzCore
  -framework VideoToolbox
)

SDK_PATH="$(xcrun --show-sdk-path)"
echo "==> Using macOS SDK: $SDK_PATH"
mkdir -p "$BUILD_DIR" "$MACOS_DIR" "$RES_DIR"

echo "==> Compiling arm64 slice (Apple Silicon M2 / M-series)..."
swiftc -O -swift-version 5 \
  -target arm64-apple-macosx14.0 \
  -sdk "$SDK_PATH" \
  "${FRAMEWORKS[@]}" \
  "${SOURCES[@]}" \
  -o "$BUILD_DIR/CM716FX-arm64"

echo "==> Compiling x86_64 slice (Intel iMac Pro Vega 56)..."
swiftc -O -swift-version 5 \
  -target x86_64-apple-macosx14.0 \
  -sdk "$SDK_PATH" \
  "${FRAMEWORKS[@]}" \
  "${SOURCES[@]}" \
  -o "$BUILD_DIR/CM716FX-x86_64"

echo "==> Creating Universal 2 binary (arm64 + x86_64)..."
lipo -create "$BUILD_DIR/CM716FX-arm64" "$BUILD_DIR/CM716FX-x86_64" -output "$MACOS_DIR/CM716FX"
lipo -info "$MACOS_DIR/CM716FX"

echo "==> Bundling Pipelines.metal, AppIcon.icns, and Info.plist..."
cp "$ROOT_DIR/Sources/CM716FX/Shaders/Pipelines.metal" "$RES_DIR/Pipelines.metal"
if [ -f "$ROOT_DIR/assets/AppIcon.icns" ]; then
  cp "$ROOT_DIR/assets/AppIcon.icns" "$RES_DIR/AppIcon.icns"
fi

cat > "$CONTENTS/Info.plist" << 'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>zh_CN</string>
    <key>CFBundleExecutable</key>
    <string>CM716FX</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>CFBundleIdentifier</key>
    <string>com.cm716fx.app</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>CM716FX</string>
    <key>CFBundleDisplayName</key>
    <string>CM716FX</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>LSMinimumSystemVersion</key>
    <string>14.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSSupportsAutomaticGraphicsSwitching</key>
    <true/>
    <key>NSCameraUsageDescription</key>
    <string>CM716FX 需要访问外置 UVC 采集卡（如绿联 CM716）以采集 Xbox 游戏画面。</string>
    <key>NSCameraUseContinuityCameraDeviceType</key>
    <true/>
    <key>NSMicrophoneUsageDescription</key>
    <string>CM716FX 需要访问采集卡 UAC 音频输入以实现低延迟 HDMI 声音直通。</string>
</dict>
</plist>
PLIST

echo "==> Ad-hoc code signing CM716FX.app..."
codesign --force --deep --sign - "$APP_BUNDLE"

echo "==> Build complete: $APP_BUNDLE"
echo "==> Running GPU headless self-test..."
"$MACOS_DIR/CM716FX" --self-test
