# Contributing to CM716FX

欢迎一起把 **CM716FX** 打造成 macOS 上面向 **GTA 6 与 30fps 主机游戏（Xbox Series S/X、PS5、Switch）** 的最强开源超分 + AI 帧生成采集卡引擎！

## 急需社区共建的方向 (Roadmap)

1. **更多主流采集卡实机 Profile 与精确 `CMTime` 适配**
   - 绿联 CM716 (`25854` USB 2.0) — 已实机适配 ✅
   - MacroSilicon MS2130 / MS2131 (USB 3.0 5Gbps 无压缩 YUY2/NV12)
   - Elgato 4K X / 4K S / HD60 X
   - AVerMedia GC553G2 / Genki ShadowCast 2 Pro
2. **神经网络超分与插帧模型扩展**
   - CoreML ANE/GPU 版 `RIFE v4.25` (FP16 `.mlpackage`) 供 macOS 14/15 Apple Silicon 用户使用
   - Metal 着色器版 `FSRCNNX` / `ArtCNN` 实时 CNN 超分（弥补 XSS 动态 720p 的模糊感）
3. **GTA 6 专项画质与 HUD 保护调优**
   - 针对 GTA 6 实机小地图/准星/车速表区域的可配置 HUD 静态蒙版矩形
   - 动态 24–30fps 掉帧场景下的自适应相位平滑算法（Adaptive Phase Pacing）
4. **低延迟音频与 HDR Tone-Mapping**
   - HDMI 10-bit HDR10 PQ → SDR 实时色调映射着色器（支持 4K HDR 采集卡）

## 提交规范

1. Fork 本仓库并新建特性分支：`git checkout -b feat/your-feature`
2. 改动后在本机运行 `./build_app.sh`，确保 `arm64` + `x86_64` 双架构编译零错误且 `--self-test` 全项通过。
3. 提交 PR 时请附上你的 Mac 型号（如 M2 / M3 / Intel）、macOS 版本、采集卡型号与 HUD 性能截图。
