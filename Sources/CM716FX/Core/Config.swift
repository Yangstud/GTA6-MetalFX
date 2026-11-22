import Foundation

/// Active frame generation backend.
public enum FrameGenBackend: String, CaseIterable, Codable {
    case off = "OFF (直通参考)"
    case auto = "AUTO (优先 Apple AI)"
    case vtLowLatency2x = "Apple VT AI 2x (30→60)"
    case vtLowLatency4x = "Apple VT AI 4x (30→120)"
    case vtCombo4K60 = "Apple VT AI 4K60 联合(超分+插帧)"
    case vtFrameRateConv = "Apple VT FRC (macOS 15.4+)"
    case metalOpticalFlow = "Metal 双向金字塔光流 (通用)"

    public var isGenerating: Bool {
        self != .off
    }
}

/// Active super-resolution / upscaling backend.
public enum UpscalerBackend: String, CaseIterable, Codable {
    case auto = "AUTO (优先 MetalFX)"
    case metalFXSpatial = "Apple MetalFX 空间超分"
    case vtLowLatencySR = "Apple VT AI 低延迟超分"
    case casLanczos = "FidelityFX CAS + 双三次超分"
    case off = "OFF (原生缩放)"
}

/// CM716 (MS2130) UVC capture profile.
public enum CaptureProfile: String, CaseIterable, Codable {
    /// Captures 1080p@60Hz in NV12/YUY2 and uses GPU MAD deduplication to strip 60Hz container duplicates.
    /// Best for GTA 6 on XSS because 24–29fps dips won't collide with 30Hz capture beat-frequency.
    case hz60NV12Dedup = "1080p60 NV12/YUY2 + GPU去重 (GTA6推荐)"
    /// Forces CM716 into native uncompressed 1080p@30fps YUY2 (4:2:2). Lowest USB bandwidth & purest chroma.
    case hz30YUY2Lossless = "1080p30 YUY2 无损直采 (稳30帧/Adrenaline伴侣)"
}

/// HDMI color range handling (consoles commonly output Limited 16-235).
public enum HDMIColorRange: String, CaseIterable, Codable {
    case limited = "Limited (16-235 标准主机)"
    case full = "Full (0-255 PC/RGB)"
}

/// One-click curated presets for CM716 + XSS GTA 6.
public enum AppPreset: String, CaseIterable {
    case m2Balanced60 = "1. GTA6 · M2 推荐 (AI 2x + MetalFX)"
    case m2Combo4K60 = "2. GTA6 · M2 4K60 联合AI (macOS 26+)"
    case m2ProMotion120 = "3. GTA6 · 120Hz 高刷 (AI 4x + MetalFX)"
    case adrenalineCleanFeed = "4. Adrenaline 纯净30fps喂帧伴侣"
    case rawPassthrough = "5. 原始采集对比 (关插帧/关超分)"
}

/// Mutable runtime configuration synchronized between UI and GPU pipeline.
public struct PipelineConfig: Codable, Equatable {
    public var frameGen: FrameGenBackend = .auto
    public var upscaler: UpscalerBackend = .auto
    public var captureProfile: CaptureProfile = .hz60NV12Dedup
    public var colorRange: HDMIColorRange = .limited
    public var enableDedup: Bool = true
    public var enableSceneCutBypass: Bool = true
    public var adrenalineCompanionMode: Bool = false
    /// Global mean absolute luma difference below which a frame is considered a candidate duplicate.
    public var dedupGlobalThreshold: Float = 0.0018
    /// Peak per-block MAD below which a frame is confirmed duplicate (preserves small moving cars/crosshairs).
    public var dedupPeakBlockThreshold: Float = 0.0085
    /// Global MAD above which a hard scene cut is detected and interpolation is bypassed for 1 interval.
    public var sceneCutThreshold: Float = 0.18
    /// Post-upscale FidelityFX CAS micro-sharpening strength (0.0...1.0) to counter XSS 720p TAA blur.
    public var casSharpness: Float = 0.28
    /// Static HUD protection sensitivity for Metal Optical Flow (0.0 = off, 1.0 = aggressive).
    public var hudProtection: Float = 0.75
    /// Target output multiplier for Metal Optical Flow (2 = 60Hz, 4 = 120Hz).
    public var opticalFlowMultiplier: Int = 2
    /// Enables VSync display pacing (recommended ON for smooth frame delivery).
    public var vsyncEnabled: Bool = true
    /// Enables built-in 1080p 30fps synthetic GTA-style motion test pattern (no capture card required).
    public var syntheticTestMode: Bool = false
    /// Simulates Xbox 60Hz HDMI duplicate frames + 25fps dips inside synthetic test mode.
    public var syntheticSimulate60HzDups: Bool = true
    /// HDMI audio passthrough mute toggle.
    public var audioMuted: Bool = false
    /// HDMI audio output volume (0.0...1.0).
    public var audioVolume: Float = 1.0
    /// Show translucent HUD overlay (auto-hidden in Adrenaline Companion Mode).
    public var showHUD: Bool = true

    public init() {}

    public mutating func apply(preset: AppPreset) {
        switch preset {
        case .m2Balanced60:
            adrenalineCompanionMode = false
            captureProfile = .hz60NV12Dedup
            enableDedup = true
            frameGen = .auto
            upscaler = .auto
            casSharpness = 0.28
            opticalFlowMultiplier = 2
            showHUD = true
        case .m2Combo4K60:
            adrenalineCompanionMode = false
            captureProfile = .hz60NV12Dedup
            enableDedup = true
            frameGen = .vtCombo4K60
            upscaler = .metalFXSpatial
            casSharpness = 0.20
            opticalFlowMultiplier = 2
            showHUD = true
        case .m2ProMotion120:
            adrenalineCompanionMode = false
            captureProfile = .hz60NV12Dedup
            enableDedup = true
            frameGen = .vtLowLatency4x
            upscaler = .auto
            casSharpness = 0.25
            opticalFlowMultiplier = 4
            showHUD = true
        case .adrenalineCleanFeed:
            adrenalineCompanionMode = true
            captureProfile = .hz60NV12Dedup
            enableDedup = true
            frameGen = .off
            upscaler = .off
            casSharpness = 0.0
            showHUD = false
        case .rawPassthrough:
            adrenalineCompanionMode = false
            captureProfile = .hz60NV12Dedup
            enableDedup = false
            frameGen = .off
            upscaler = .off
            casSharpness = 0.0
            showHUD = true
        }
    }
}

/// Real-time telemetry snapshot displayed in the HUD and logged during self-test.
public struct TelemetrySnapshot {
    public var deviceName: String = "未检测到采集卡 (按 T 启用内置 30fps 测试源)"
    public var pixelFormatFourCC: String = "----"
    public var inputResolution: String = "0x0"
    public var outputResolution: String = "0x0"
    public var usbBandwidthWarning: String? = nil

    public var captureFPS: Double = 0
    public var uniqueFPS: Double = 0
    public var outputFPS: Double = 0
    public var droppedDuplicateRatio: Double = 0
    public var sceneCutCount: Int = 0

    public var gpuConvertDedupMs: Double = 0
    public var gpuFrameGenMs: Double = 0
    public var gpuUpscalePresentMs: Double = 0
    public var totalPipelineMs: Double = 0

    public var activeFrameGenLabel: String = "初始化中..."
    public var activeUpscalerLabel: String = "初始化中..."
    public var hardwareCapabilitiesSummary: String = ""
    public var lastGlobalMAD: Float = 0
    public var lastPeakBlockMAD: Float = 0

    public init() {}
}
