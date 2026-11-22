#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo " 绿联 CM716 (MS2130) USB 速率与设备体检"
echo "=========================================="
echo ""
echo "[1] 检查 USB 总线协商速率与硬件规格："
echo "    说明：绿联 CM716（料号 25854）实测为原生 USB 2.0（bcdUSB 2.0 / 480 Mb/s）方案，直插显示 Up to 480 Mb/s 即为满速；"
echo "    建议在 CM716FX 中对比 1080p60（延迟低）与按 C 切换 1080p30（单帧码率更高、画质更干净）。"
system_profiler SPUSBDataType 2>/dev/null | grep -iE -B 2 -A 8 "ugreen|25854|cm716|ms213|capture|video|hdmi|macrosilicon|usb3.*video" || echo "  (当前未检测到外置 UVC 采集卡连接)"

echo ""
echo "[2] 检查本机芯片与系统版本："
echo "  Arch: $(uname -m) | macOS: $(sw_vers -productVersion) ($(sw_vers -buildVersion))"
echo "  GPU : $(system_profiler SPDisplaysDataType 2>/dev/null | grep "Chipset Model" | head -n 1 | xargs)"

echo ""
echo "[3] 检查 CM716FX.app 架构切片："
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -x "$ROOT_DIR/CM716FX.app/Contents/MacOS/CM716FX" ]; then
  lipo -info "$ROOT_DIR/CM716FX.app/Contents/MacOS/CM716FX"
else
  echo "  尚未构建 CM716FX.app，请先运行 ./build_app.sh"
fi
