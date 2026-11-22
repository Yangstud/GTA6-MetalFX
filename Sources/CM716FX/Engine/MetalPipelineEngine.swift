import CoreMedia
import CoreVideo
import Foundation
import Metal
import MetalFX
import QuartzCore

// C-compatible uniform layouts matching Pipelines.metal
private struct ColorConvertParams {
    var width: UInt32
    var height: UInt32
    var isFullRange: UInt32
    var isYUVS: UInt32
}

private struct WarpParams {
    var phase: Float
    var hudProtection: Float
    var flowScaleX: Float
    var flowScaleY: Float
}

private struct PresentParams {
    var sharpness: Float
    var useBicubic: UInt32
    var srcWidth: Float
    var srcHeight: Float
    var dstWidth: Float
    var dstHeight: Float
    var pad0: Float = 0
    var pad1: Float = 0
}

private struct SynthParams {
    var timeSec: Float
    var width: UInt32
    var height: UInt32
    var sceneVariant: UInt32
}

/// Holds a ready-to-display frame plus optional backing CVPixelBuffer/CVMetalTexture lifetimes.
private final class PacedFrame {
    let texture: MTLTexture
    let backingBuffer: CVPixelBuffer?
    let backingMetalTex: CVMetalTexture?
    let pts: CMTime
    let isInterpolated: Bool

    init(
        texture: MTLTexture,
        backingBuffer: CVPixelBuffer? = nil,
        backingMetalTex: CVMetalTexture? = nil,
        pts: CMTime = .invalid,
        isInterpolated: Bool = false
    ) {
        self.texture = texture
        self.backingBuffer = backingBuffer
        self.backingMetalTex = backingMetalTex
        self.pts = pts
        self.isInterpolated = isInterpolated
    }
}

/// Complete zero-copy GPU pipeline:
/// CM716 UVC (NV12/YUY2) -> BT.709 RGB -> GPU Dedup & Scene-Cut ->
/// [Apple VT Low-Latency AI FG / VT FRC / Metal Pyramid Bidirectional Optical Flow] ->
/// Adaptive VSync Pacer -> [Apple MetalFX Spatial Scaler / VT AI SR / FidelityFX CAS] -> CAMetalLayer
public final class MetalPipelineEngine {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    public let capabilities: AppleHardwareCapabilities

    private let library: MTLLibrary
    private var textureCache: CVMetalTextureCache?

    // Compute pipelines
    private let nv12ToBGRAPSO: MTLComputePipelineState
    private let yuy2ToBGRAPSO: MTLComputePipelineState
    private let copyBGRAPSO: MTLComputePipelineState
    private let bgraToNV12PSO: MTLComputePipelineState
    private let dedupMADPSO: MTLComputePipelineState
    private let rgbToLumaHalfPSO: MTLComputePipelineState
    private let lumaDown2xPSO: MTLComputePipelineState
    private let flowCoarsePSO: MTLComputePipelineState
    private let flowRefinePSO: MTLComputePipelineState
    private let flowSmoothPSO: MTLComputePipelineState
    private let flowWarpPSO: MTLComputePipelineState
    private let synthPatternPSO: MTLComputePipelineState
    private let presentRenderPSO: MTLRenderPipelineState

    // VideoToolbox ML backend (macOS 26+ LowLatencyFG/SR, macOS 15.4+ FRC)
    private let vtEngine = VTFrameGenEngine()

    // MetalFX Spatial Scaler (Apple Silicon)
    private var metalFXScalerBox: AnyObject?
    private var metalFXInSize: (Int, Int) = (0, 0)
    private var metalFXOutSize: (Int, Int) = (0, 0)
    private var metalFXOutTex: MTLTexture?

    // Working GPU resources (allocated lazily for active input dimensions)
    private var inputWidth: Int = 0
    private var inputHeight: Int = 0
    private var stagingBGRA: MTLTexture?
    private var uniqueRing: [MTLTexture] = []
    private var uniqueRingIndex: Int = 0
    private var prevUniqueTex: MTLTexture?
    private var prevUniqueVTBuffer: CVPixelBuffer?
    private var prevUniquePTS: CMTime = .invalid

    // Metal Optical Flow pyramids (A = prev, B = curr) + 3 interpolated output textures (up to 4x / 120Hz)
    private var lumaA: [MTLTexture] = []   // [W/2, W/4, W/8]
    private var lumaB: [MTLTexture] = []
    private var flowAB: [MTLTexture] = []  // [W/2, W/4, W/8]
    private var flowBA: [MTLTexture] = []
    private var flowSmoothAB: MTLTexture?
    private var flowSmoothBA: MTLTexture?
    private var ofOutputRing: [[MTLTexture]] = [] // Double-buffered sets of 3 interpolated textures
    private var ofOutputBank: Int = 0

    // Dedup readback buffer (16x9 = 144 floats = 576 bytes, shared memory)
    private let dedupGridW = 16
    private let dedupGridH = 9
    private let dedupBuffer: MTLBuffer

    // Synchronization & Pacing State
    private let configLock = NSLock()
    private var _config = PipelineConfig()
    public var config: PipelineConfig {
        get { configLock.lock(); defer { configLock.unlock() }; return _config }
        set { configLock.lock(); _config = newValue; configLock.unlock() }
    }

    private let pacerLock = NSLock()
    private var pacedSequence: [PacedFrame] = []
    private var pacedIndex: Int = 0
    private var lastPresentedFrame: PacedFrame?

    private let ingestQueue = DispatchQueue(label: "com.cm716fx.gpu.ingest", qos: .userInteractive)

    // Telemetry tracking
    private let statsLock = NSLock()
    private var _telemetry = TelemetrySnapshot()
    private var windowStartTime: CFAbsoluteTime = CFAbsoluteTimeGetCurrent()
    private var countCaptured: Int = 0
    private var countUnique: Int = 0
    private var countDroppedDup: Int = 0
    private var countOutput: Int = 0
    private var countSceneCuts: Int = 0

    // Synthetic test pattern timer state
    private var synthTimer: DispatchSourceTimer?
    private var synthStartTime: CFAbsoluteTime = 0
    private var synthTickIndex: Int = 0
    private var synthSourceTex: MTLTexture?
    private var synthSourcePB: CVPixelBuffer?

    public init(shaderSourcePath: String? = nil) throws {
        guard let dev = MTLCreateSystemDefaultDevice(),
              let queue = dev.makeCommandQueue() else {
            throw NSError(domain: "CM716FX", code: -1, userInfo: [NSLocalizedDescriptionKey: "无法初始化 Metal GPU 设备"])
        }
        self.device = dev
        self.commandQueue = queue
        self.capabilities = VTFrameGenEngine.probeCapabilities(device: dev)

        // Load Pipelines.metal source from Bundle / explicit path / sibling path
        let source = try Self.loadMetalSource(explicitPath: shaderSourcePath)
        let compileOpts = MTLCompileOptions()
        compileOpts.fastMathEnabled = true
        let lib = try dev.makeLibrary(source: source, options: compileOpts)
        self.library = lib

        func makeCompute(_ name: String) throws -> MTLComputePipelineState {
            guard let fn = lib.makeFunction(name: name) else {
                throw NSError(domain: "CM716FX", code: -2, userInfo: [NSLocalizedDescriptionKey: "缺少 Metal 内核函数: \(name)"])
            }
            return try dev.makeComputePipelineState(function: fn)
        }

        self.nv12ToBGRAPSO    = try makeCompute("nv12_to_bgra_bt709")
        self.yuy2ToBGRAPSO    = try makeCompute("yuy2_to_bgra_bt709")
        self.copyBGRAPSO      = try makeCompute("copy_bgra")
        self.bgraToNV12PSO    = try makeCompute("bgra_to_nv12_bt709")
        self.dedupMADPSO      = try makeCompute("frame_dedup_mad")
        self.rgbToLumaHalfPSO = try makeCompute("rgb_to_luma_half")
        self.lumaDown2xPSO    = try makeCompute("luma_downsample_2x")
        self.flowCoarsePSO    = try makeCompute("optical_flow_coarse")
        self.flowRefinePSO    = try makeCompute("optical_flow_refine")
        self.flowSmoothPSO    = try makeCompute("optical_flow_smooth")
        self.flowWarpPSO      = try makeCompute("optical_flow_warp_blend")
        self.synthPatternPSO  = try makeCompute("synthetic_test_pattern")

        guard let vs = lib.makeFunction(name: "fullscreen_quad_vs"),
              let fs = lib.makeFunction(name: "present_cas_fs") else {
            throw NSError(domain: "CM716FX", code: -3, userInfo: [NSLocalizedDescriptionKey: "缺少呈现渲染管线着色器"])
        }
        let rpd = MTLRenderPipelineDescriptor()
        rpd.vertexFunction = vs
        rpd.fragmentFunction = fs
        rpd.colorAttachments[0].pixelFormat = .bgra8Unorm
        self.presentRenderPSO = try dev.makeRenderPipelineState(descriptor: rpd)

        guard let dbuf = dev.makeBuffer(length: dedupGridW * dedupGridH * MemoryLayout<Float>.stride, options: .storageModeShared) else {
            throw NSError(domain: "CM716FX", code: -4, userInfo: [NSLocalizedDescriptionKey: "无法分配 Dedup GPU 缓冲"])
        }
        self.dedupBuffer = dbuf

        CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, dev, nil, &self.textureCache)

        self._telemetry.hardwareCapabilitiesSummary = capabilities.summaryString
    }

    private static func loadMetalSource(explicitPath: String?) throws -> String {
        var candidates: [String] = []
        if let p = explicitPath { candidates.append(p) }
        if let resURL = Bundle.main.url(forResource: "Pipelines", withExtension: "metal") {
            candidates.append(resURL.path)
        }
        let execDir = (Bundle.main.executablePath as NSString?)?.deletingLastPathComponent ?? FileManager.default.currentDirectoryPath
        candidates.append((execDir as NSString).appendingPathComponent("Pipelines.metal"))
        candidates.append((execDir as NSString).appendingPathComponent("../Resources/Pipelines.metal"))
        candidates.append(FileManager.default.currentDirectoryPath + "/Sources/CM716FX/Shaders/Pipelines.metal")

        for path in candidates {
            if FileManager.default.fileExists(atPath: path),
               let content = try? String(contentsOfFile: path, encoding: .utf8) {
                return content
            }
        }
        throw NSError(domain: "CM716FX", code: -5, userInfo: [NSLocalizedDescriptionKey: "未找到 Pipelines.metal 着色器源码"])
    }

    // MARK: - Telemetry & External Status Updates

    public func updateCaptureStatus(deviceName: String, fourCC: String, resolution: String, warning: String?) {
        statsLock.lock()
        _telemetry.deviceName = deviceName
        _telemetry.pixelFormatFourCC = fourCC
        _telemetry.inputResolution = resolution
        _telemetry.usbBandwidthWarning = warning
        statsLock.unlock()
    }

    public func telemetrySnapshot() -> TelemetrySnapshot {
        statsLock.lock()
        defer { statsLock.unlock() }
        let now = CFAbsoluteTimeGetCurrent()
        let elapsed = now - windowStartTime
        if elapsed >= 0.5 {
            _telemetry.captureFPS = Double(countCaptured) / elapsed
            _telemetry.uniqueFPS = Double(countUnique) / elapsed
            _telemetry.outputFPS = Double(countOutput) / elapsed
            _telemetry.droppedDuplicateRatio = countCaptured > 0 ? (Double(countDroppedDup) / Double(countCaptured)) : 0
            _telemetry.sceneCutCount = countSceneCuts
            countCaptured = 0
            countUnique = 0
            countDroppedDup = 0
            countOutput = 0
            windowStartTime = now
        }
        return _telemetry
    }

    // MARK: - Built-in Synthetic 30fps Test Source (Simulates CM716 60Hz Container + Duplicate Frames)

    public func setSyntheticTestMode(enabled: Bool) {
        if enabled {
            startSyntheticGenerator()
        } else {
            stopSyntheticGenerator()
        }
    }

    private func startSyntheticGenerator() {
        stopSyntheticGenerator()
        synthStartTime = CFAbsoluteTimeGetCurrent()
        synthTickIndex = 0
        updateCaptureStatus(
            deviceName: "内置 1080p 30fps 模拟源 (模拟 CM716 60Hz 重复帧)",
            fourCC: "SYNTH",
            resolution: "1920x1080@60(30u)",
            warning: nil
        )

        let timer = DispatchSource.makeTimerSource(queue: ingestQueue)
        // Fire at 60Hz (16.667ms) just like CM716 HDMI 60Hz output carrying a 30fps Xbox game!
        timer.schedule(deadline: .now(), repeating: .microseconds(16_667), leeway: .microseconds(500))
        timer.setEventHandler { [weak self] in
            guard let self else { return }
            self.generateAndIngestSyntheticFrame()
        }
        synthTimer = timer
        timer.resume()
    }

    private func stopSyntheticGenerator() {
        synthTimer?.cancel()
        synthTimer = nil
    }

    private func generateAndIngestSyntheticFrame() {
        let w = 1920
        let h = 1080
        ensureResources(width: w, height: h)

        if synthSourcePB == nil {
            let attrs: [String: Any] = [
                kCVPixelBufferWidthKey as String: w,
                kCVPixelBufferHeightKey as String: h,
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferMetalCompatibilityKey as String: true,
                kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any]
            ]
            var pb: CVPixelBuffer?
            CVPixelBufferCreate(kCFAllocatorDefault, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb)
            synthSourcePB = pb
        }
        guard let pb = synthSourcePB,
              let cvTex = wrapPlaneTexture(pb, format: .bgra8Unorm, width: w, height: h, planeIndex: 0),
              let dstTex = CVMetalTextureGetTexture(cvTex) else {
            return
        }

        let cfg = self.config
        let tick = synthTickIndex
        synthTickIndex += 1

        // Simulate 30fps game inside a 60Hz HDMI container: update game time only on even ticks (A, A, B, B...)
        // Plus occasionally drop to 20fps (hold 3 ticks every ~3 seconds) like heavy GTA 6 gunfights!
        let isSimulatedDup: Bool
        if cfg.syntheticSimulate60HzDups {
            let heavyDipHold = (tick % 180 == 62) // occasional 3-tick hold simulating 20fps dip
            isSimulatedDup = (tick % 2 == 1) || heavyDipHold
        } else {
            isSimulatedDup = false
        }

        if !isSimulatedDup {
            let t = Float(CFAbsoluteTimeGetCurrent() - synthStartTime)
            // Switch scene variant every 8 seconds to exercise Scene-Cut Bypass!
            let variant = UInt32(Int(t / 8.0))
            if let cmdBuf = commandQueue.makeCommandBuffer(),
               let enc = cmdBuf.makeComputeCommandEncoder() {
                enc.setComputePipelineState(synthPatternPSO)
                enc.setTexture(dstTex, index: 0)
                var sp = SynthParams(timeSec: t, width: UInt32(w), height: UInt32(h), sceneVariant: variant)
                enc.setBytes(&sp, length: MemoryLayout<SynthParams>.stride, index: 0)
                dispatch2D(enc, width: w, height: h, pso: synthPatternPSO)
                enc.endEncoding()
                cmdBuf.commit()
                cmdBuf.waitUntilCompleted()
            }
        }

        let pts = CMTime(seconds: CFAbsoluteTimeGetCurrent() - synthStartTime, preferredTimescale: 60000)
        processFrameOnIngestQueue(pixelBuffer: pb, pts: pts, fourCC: kCVPixelFormatType_32BGRA)
    }

    // MARK: - Live Capture Card Ingestion

    /// Called by `CM716CaptureManager` on its real-time video queue.
    public func ingestCaptureBuffer(_ pixelBuffer: CVPixelBuffer, pts: CMTime, fourCC: OSType) {
        if config.syntheticTestMode { return }
        // Run synchronously on the single-threaded video queue (alwaysDiscardsLateVideoFrames keeps latency minimal)
        processFrameOnIngestQueue(pixelBuffer: pixelBuffer, pts: pts, fourCC: fourCC)
    }

    private func processFrameOnIngestQueue(pixelBuffer: CVPixelBuffer, pts: CMTime, fourCC: OSType) {
        let t0 = CFAbsoluteTimeGetCurrent()
        let w = CVPixelBufferGetWidth(pixelBuffer)
        let h = CVPixelBufferGetHeight(pixelBuffer)
        guard w >= 320 && h >= 180 else { return }

        ensureResources(width: w, height: h)
        let cfg = self.config

        statsLock.lock()
        countCaptured += 1
        statsLock.unlock()

        // 1. Zero-copy YUV/BGRA -> BT.709 stagingBGRA + GPU Duplicate-Frame MAD check
        guard let staging = stagingBGRA,
              let cmd1 = commandQueue.makeCommandBuffer() else { return }

        encodeColorConvert(cmdBuf: cmd1, pixelBuffer: pixelBuffer, fourCC: fourCC, colorRange: cfg.colorRange, outTex: staging)

        let runDedup = cfg.enableDedup && (prevUniqueTex != nil)
        if runDedup, let prevTex = prevUniqueTex, let enc = cmd1.makeComputeCommandEncoder() {
            enc.setComputePipelineState(dedupMADPSO)
            enc.setTexture(staging, index: 0)
            enc.setTexture(prevTex, index: 1)
            enc.setBuffer(dedupBuffer, offset: 0, index: 0)
            enc.dispatchThreadgroups(
                MTLSize(width: dedupGridW, height: dedupGridH, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1)
            )
            enc.endEncoding()
        }

        cmd1.commit()
        cmd1.waitUntilCompleted()
        let tAfterConvertDedup = CFAbsoluteTimeGetCurrent()

        var isSceneCut = false
        var globalMAD: Float = 0
        var peakBlockMAD: Float = 0

        if runDedup {
            let ptr = dedupBuffer.contents().bindMemory(to: Float.self, capacity: dedupGridW * dedupGridH)
            let totalBlocks = dedupGridW * dedupGridH
            var sum: Float = 0
            for i in 0..<totalBlocks {
                let v = ptr[i]
                sum += v
                if v > peakBlockMAD { peakBlockMAD = v }
            }
            globalMAD = sum / Float(totalBlocks)

            statsLock.lock()
            _telemetry.lastGlobalMAD = globalMAD
            _telemetry.lastPeakBlockMAD = peakBlockMAD
            _telemetry.gpuConvertDedupMs = (tAfterConvertDedup - t0) * 1000.0
            statsLock.unlock()

            // Duplicate HDMI frame detection: both whole-frame MAD and max local block MAD must be quiet!
            if globalMAD < cfg.dedupGlobalThreshold && peakBlockMAD < cfg.dedupPeakBlockThreshold {
                statsLock.lock()
                countDroppedDup += 1
                statsLock.unlock()
                return
            }

            if cfg.enableSceneCutBypass && globalMAD > cfg.sceneCutThreshold {
                isSceneCut = true
                statsLock.lock()
                countSceneCuts += 1
                _telemetry.sceneCutCount = countSceneCuts
                statsLock.unlock()
            }
        } else {
            statsLock.lock()
            _telemetry.gpuConvertDedupMs = (tAfterConvertDedup - t0) * 1000.0
            statsLock.unlock()
        }

        // 2. Promote stagingBGRA into our persistent unique frame ring slot
        statsLock.lock()
        countUnique += 1
        statsLock.unlock()

        let currUniqueTex = uniqueRing[uniqueRingIndex]
        uniqueRingIndex = (uniqueRingIndex + 1) % uniqueRing.count

        // Prepare VideoToolbox ML session if requested (non-blocking warmup)
        let desiredVTKind = resolveVTKind(for: cfg)
        if desiredVTKind != .none {
            vtEngine.prepareSessionAsync(kind: desiredVTKind, width: w, height: h)
        }
        if cfg.upscaler == .vtLowLatencySR && capabilities.supportsVTLowLatencySR {
            vtEngine.prepareSuperResAsync(width: w, height: h)
        }

        // Allocate optional VT-compatible IOSurface source pixel buffer if VT is active/warming
        let currVTBuffer: CVPixelBuffer? = (desiredVTKind != .none && vtEngine.isReady)
            ? vtEngine.allocateSourceBuffer()
            : nil

        let tFGStart = CFAbsoluteTimeGetCurrent()
        var newPaced: [PacedFrame] = []
        var activeFGName = "OFF (原始直通)"

        guard let cmd2 = commandQueue.makeCommandBuffer() else { return }

        // Copy staging -> currUniqueTex (and optionally populate currVTBuffer on the exact same command buffer!)
        if let enc = cmd2.makeComputeCommandEncoder() {
            enc.setComputePipelineState(copyBGRAPSO)
            enc.setTexture(staging, index: 0)
            enc.setTexture(currUniqueTex, index: 1)
            dispatch2D(enc, width: w, height: h, pso: copyBGRAPSO)
            enc.endEncoding()
        }
        if let vtPB = currVTBuffer {
            encodeWriteToVTBuffer(cmdBuf: cmd2, srcBGRA: staging, dstVTBuffer: vtPB, width: w, height: h)
        }

        // 3. Frame Generation Stage
        let shouldGenerate = cfg.frameGen.isGenerating && !cfg.adrenalineCompanionMode && !isSceneCut && (prevUniqueTex != nil)

        if shouldGenerate, let prevTex = prevUniqueTex {
            var usedVT = false

            // 3A. Try Apple VideoToolbox ML Engine (macOS 26 LowLatencyFG / 4K60 Combo / macOS 15.4 FRC)
            if desiredVTKind != .none,
               vtEngine.isReady,
               let prevPB = prevUniqueVTBuffer,
               let currPB = currVTBuffer {
                if let vtOut = vtEngine.encodeInterpolation(
                    commandBuffer: cmd2,
                    previousBuffer: prevPB,
                    previousPTS: prevUniquePTS,
                    currentBuffer: currPB,
                    currentPTS: pts,
                    isSceneCut: isSceneCut
                ) {
                    usedVT = true
                    switch desiredVTKind {
                    case .lowLatencyTemporal2x:     activeFGName = "Apple VT Low-Latency AI 2x"
                    case .lowLatencyTemporal4x:     activeFGName = "Apple VT Low-Latency AI 4x"
                    case .lowLatencySpatial2xCombo: activeFGName = "Apple VT AI 4K60 联合(2x超分+2x插帧)"
                    case .frameRateConversion2x:    activeFGName = "Apple VT FRC ML 2x"
                    case .none: break
                    }

                    for midPB in vtOut.intermediateBuffers {
                        if let paced = wrapVTOutputToPacedFrame(cmdBuf: cmd2, pb: midPB, pts: pts, isInterpolated: true) {
                            newPaced.append(paced)
                        }
                    }
                    if let scaledSrcPB = vtOut.upscaledSourceBuffer,
                       let pacedSrc = wrapVTOutputToPacedFrame(cmdBuf: cmd2, pb: scaledSrcPB, pts: pts, isInterpolated: false) {
                        newPaced.append(pacedSrc)
                    } else {
                        newPaced.append(PacedFrame(texture: currUniqueTex, backingBuffer: currVTBuffer, pts: pts, isInterpolated: false))
                    }
                }
            }

            // 3B. Universal Metal Pyramid Bidirectional Optical Flow + Occlusion + HUD Guard
            if !usedVT {
                let phases: [Float] = (cfg.frameGen == .vtLowLatency4x || cfg.opticalFlowMultiplier >= 4)
                    ? [0.25, 0.50, 0.75]
                    : [0.50]
                let interpTextures = encodeMetalOpticalFlow(
                    cmdBuf: cmd2,
                    frameA: prevTex,
                    frameB: currUniqueTex,
                    phases: phases,
                    hudProtection: cfg.hudProtection
                )
                let multTag = phases.count == 3 ? "4x (30→120)" : "2x (30→60)"
                activeFGName = (desiredVTKind != .none && !vtEngine.isReady)
                    ? "Metal 光流 \(multTag) [VT AI 预热中]"
                    : "Metal 双向金字塔光流 \(multTag)"

                for tex in interpTextures {
                    newPaced.append(PacedFrame(texture: tex, pts: pts, isInterpolated: true))
                }
                newPaced.append(PacedFrame(texture: currUniqueTex, backingBuffer: currVTBuffer, pts: pts, isInterpolated: false))
            }
        } else {
            if cfg.adrenalineCompanionMode {
                activeFGName = "Adrenaline 纯净喂帧模式 (30fps 去重直通)"
            } else if isSceneCut {
                activeFGName = "场景硬切保护 (跳过插帧)"
            }
            newPaced.append(PacedFrame(texture: currUniqueTex, backingBuffer: currVTBuffer, pts: pts, isInterpolated: false))
        }

        cmd2.commit()
        cmd2.waitUntilCompleted()

        let tFGEnd = CFAbsoluteTimeGetCurrent()
        prevUniqueTex = currUniqueTex
        prevUniqueVTBuffer = currVTBuffer
        prevUniquePTS = pts

        // Publish fresh paced sequence (strict max depth = 1 step -> ZERO cumulative lag!)
        pacerLock.lock()
        pacedSequence = newPaced
        pacedIndex = 0
        pacerLock.unlock()

        statsLock.lock()
        _telemetry.gpuFrameGenMs = (tFGEnd - tFGStart) * 1000.0
        _telemetry.activeFrameGenLabel = activeFGName
        statsLock.unlock()
    }

    // MARK: - VSync Display Pacing & MetalFX Spatial Upscaling Render

    /// Invoked on every display VSync tick (`CVDisplayLink`) to present the next paced frame with zero lag drift.
    public func renderNextDisplayFrame(into drawable: CAMetalDrawable, viewportSize: CGSize) {
        let t0 = CFAbsoluteTimeGetCurrent()
        let frameToDraw: PacedFrame?

        pacerLock.lock()
        if pacedIndex < pacedSequence.count {
            frameToDraw = pacedSequence[pacedIndex]
            pacedIndex += 1
            lastPresentedFrame = frameToDraw
        } else {
            frameToDraw = lastPresentedFrame
        }
        pacerLock.unlock()

        guard let frame = frameToDraw,
              let cmdBuf = commandQueue.makeCommandBuffer() else {
            return
        }

        let cfg = self.config
        let dstW = drawable.texture.width
        let dstH = drawable.texture.height
        let srcW = frame.texture.width
        let srcH = frame.texture.height

        var sourceForPresent: MTLTexture = frame.texture
        var useBicubic: UInt32 = 0
        var activeUpscalerName = "原生直出"

        // Calculate aspect-fitted MetalFX target size matching Retina drawable (capped at 4K 3840x2160 for optimal 60/120Hz headroom on 5K displays)
        let rawFit = aspectFittedSize(srcW: srcW, srcH: srcH, dstW: dstW, dstH: dstH)
        let (fitW, fitH) = aspectFittedSize(srcW: srcW, srcH: srcH, dstW: min(rawFit.0, 3840), dstH: min(rawFit.1, 2160))

        if !cfg.adrenalineCompanionMode {
            let wantMetalFX = (cfg.upscaler == .auto || cfg.upscaler == .metalFXSpatial)
                && capabilities.supportsMetalFXSpatial
                && (fitW > srcW || fitH > srcH)
            let wantVTSR = (cfg.upscaler == .vtLowLatencySR)
                && capabilities.supportsVTLowLatencySR
                && (frame.backingBuffer != nil)

            if wantVTSR, let srcPB = frame.backingBuffer,
               let srPB = vtEngine.encodeSuperResolution(commandBuffer: cmdBuf, sourceBuffer: srcPB, pts: frame.pts),
               let srWrapped = wrapVTOutputToPacedFrame(cmdBuf: cmdBuf, pb: srPB, pts: frame.pts, isInterpolated: frame.isInterpolated) {
                sourceForPresent = srWrapped.texture
                activeUpscalerName = "Apple VT AI 低延迟超分"
            } else if wantMetalFX, #available(macOS 13.0, *) {
                if let (scaler, outTex) = ensureMetalFXSpatialScaler(inW: srcW, inH: srcH, outW: fitW, outH: fitH) {
                    scaler.colorTexture = frame.texture
                    scaler.inputContentWidth = srcW
                    scaler.inputContentHeight = srcH
                    scaler.outputTexture = outTex
                    scaler.encode(commandBuffer: cmdBuf)
                    sourceForPresent = outTex
                    activeUpscalerName = cfg.casSharpness > 0.01
                        ? String(format: "Apple MetalFX 空间超分 + CAS %.0f%%", cfg.casSharpness * 100)
                        : "Apple MetalFX 空间超分"
                }
            } else if cfg.upscaler == .casLanczos || (cfg.upscaler == .auto && !capabilities.supportsMetalFXSpatial) {
                useBicubic = (fitW > srcW || fitH > srcH) ? 1 : 0
                activeUpscalerName = String(format: "Catmull-Rom + FidelityFX CAS %.0f%%", cfg.casSharpness * 100)
            } else if srcW >= 3840 {
                activeUpscalerName = "VT AI 4K 联合输出 (已 2x 超分)"
            }
        } else {
            activeUpscalerName = "OFF (Adrenaline 洁净输出)"
        }

        // Final render pass -> CAMetalDrawable (with optional FidelityFX CAS micro-sharpening)
        let rpd = MTLRenderPassDescriptor()
        rpd.colorAttachments[0].texture = drawable.texture
        rpd.colorAttachments[0].loadAction = .clear
        rpd.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        rpd.colorAttachments[0].storeAction = .store

        if let renc = cmdBuf.makeRenderCommandEncoder(descriptor: rpd) {
            renc.setRenderPipelineState(presentRenderPSO)
            renc.setFragmentTexture(sourceForPresent, index: 0)
            var pp = PresentParams(
                sharpness: cfg.adrenalineCompanionMode ? 0.0 : cfg.casSharpness,
                useBicubic: useBicubic,
                srcWidth: Float(sourceForPresent.width),
                srcHeight: Float(sourceForPresent.height),
                dstWidth: Float(dstW),
                dstHeight: Float(dstH)
            )
            renc.setFragmentBytes(&pp, length: MemoryLayout<PresentParams>.stride, index: 0)
            renc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            renc.endEncoding()
        }

        cmdBuf.present(drawable)
        cmdBuf.addCompletedHandler { [weak self] _ in
            guard let self else { return }
            let elapsedMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0
            self.statsLock.lock()
            self.countOutput += 1
            self._telemetry.outputResolution = "\(dstW)x\(dstH)"
            self._telemetry.gpuUpscalePresentMs = elapsedMs
            self._telemetry.totalPipelineMs = self._telemetry.gpuConvertDedupMs + self._telemetry.gpuFrameGenMs + elapsedMs
            self._telemetry.activeUpscalerLabel = activeUpscalerName
            self.statsLock.unlock()
        }
        cmdBuf.commit()
    }

    // MARK: - Headless GPU Self-Test & Benchmark (CLI `--self-test`)

    public func runHeadlessSelfTest(frameCount: Int = 60) -> Bool {
        print("============================================================")
        print(" CM716FX GPU Pipeline Self-Test & Benchmark")
        print(" Hardware: \(capabilities.summaryString)")
        print("============================================================")

        var cfg = PipelineConfig()
        cfg.syntheticTestMode = true
        cfg.syntheticSimulate60HzDups = true
        cfg.enableDedup = true
        cfg.enableSceneCutBypass = true
        cfg.frameGen = .auto
        cfg.upscaler = .auto
        self.config = cfg

        let w = 1920, h = 1080
        ensureResources(width: w, height: h)

        var totalConvertDedupMs = 0.0
        var totalFGMs = 0.0
        var uniqueCount = 0
        var dupDropped = 0

        let attrs: [String: Any] = [
            kCVPixelBufferWidthKey as String: w,
            kCVPixelBufferHeightKey as String: h,
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any]
        ]
        var pbOpt: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pbOpt)
        guard let pb = pbOpt,
              let cvTex = wrapPlaneTexture(pb, format: .bgra8Unorm, width: w, height: h, planeIndex: 0),
              let dstTex = CVMetalTextureGetTexture(cvTex) else {
            print("[FAIL] 无法创建测试 CVPixelBuffer")
            return false
        }

        for i in 0..<frameCount {
            let isDup = (i % 2 == 1)
            if !isDup {
                let t = Float(i) / 60.0
                let variant: UInt32 = (i >= 40) ? 1 : 0 // trigger 1 hard scene cut at frame 40!
                if let cb = commandQueue.makeCommandBuffer(),
                   let enc = cb.makeComputeCommandEncoder() {
                    enc.setComputePipelineState(synthPatternPSO)
                    enc.setTexture(dstTex, index: 0)
                    var sp = SynthParams(timeSec: t, width: UInt32(w), height: UInt32(h), sceneVariant: variant)
                    enc.setBytes(&sp, length: MemoryLayout<SynthParams>.stride, index: 0)
                    dispatch2D(enc, width: w, height: h, pso: synthPatternPSO)
                    enc.endEncoding()
                    cb.commit()
                    cb.waitUntilCompleted()
                }
            }

            let prevDrop = getDroppedDupCount()
            let prevUnique = getUniqueCount()
            let pts = CMTime(value: CMTimeValue(i), timescale: 60)
            processFrameOnIngestQueue(pixelBuffer: pb, pts: pts, fourCC: kCVPixelFormatType_32BGRA)
            let snap = rawTelemetry()

            if getDroppedDupCount() > prevDrop {
                dupDropped += 1
            } else if getUniqueCount() > prevUnique {
                uniqueCount += 1
                totalConvertDedupMs += snap.gpuConvertDedupMs
                if uniqueCount > 2 { // skip cold first-frame compile jitter
                    totalFGMs += snap.gpuFrameGenMs
                }
            }
        }

        let avgDedup = totalConvertDedupMs / Double(max(uniqueCount, 1))
        let avgFG = totalFGMs / Double(max(uniqueCount - 2, 1))
        let snap = rawTelemetry()

        // Also benchmark Apple MetalFX Spatial Upscaling (1080p -> 4K 3840x2160) on real GPU!
        var metalFXStatus = "N/A (设备不支持 MetalFX)"
        var metalFXMs = 0.0
        var metalFXOK = true
        if capabilities.supportsMetalFXSpatial, #available(macOS 13.0, *), let srcTex = prevUniqueTex {
            if let (scaler, outTex) = ensureMetalFXSpatialScaler(inW: w, inH: h, outW: 3840, outH: 2160) {
                // Warmup 2 runs + measure 10 runs
                for run in 0..<12 {
                    let ts = CFAbsoluteTimeGetCurrent()
                    if let cb = commandQueue.makeCommandBuffer() {
                        scaler.colorTexture = srcTex
                        scaler.inputContentWidth = w
                        scaler.inputContentHeight = h
                        scaler.outputTexture = outTex
                        scaler.encode(commandBuffer: cb)
                        cb.commit()
                        cb.waitUntilCompleted()
                        if run >= 2 {
                            metalFXMs += (CFAbsoluteTimeGetCurrent() - ts) * 1000.0
                        }
                    }
                }
                metalFXMs /= 10.0
                metalFXStatus = String(format: "PASS (1080p → 4K 平均 %.2f ms)", metalFXMs)
            } else {
                metalFXStatus = "FAIL (MTLFXSpatialScaler 创建失败)"
                metalFXOK = false
            }
        }

        print(String(format: "  输入测试帧 (模拟 CM716 60Hz 含50%%重复帧): %d 帧", frameCount))
        print(String(format: "  识别唯一有效帧: %d | GPU 精准剔除重复帧: %d (期望 ~%d)", uniqueCount, dupDropped, frameCount / 2))
        print(String(format: "  场景硬切保护触发次数: %d (期望 >= 1)", snap.sceneCutCount))
        print(String(format: "  当前激活插帧引擎: %@", snap.activeFrameGenLabel))
        print(String(format: "  GPU 格式转换+去重平均耗时: %.2f ms", avgDedup))
        print(String(format: "  GPU 1080p 插帧平均耗时:    %.2f ms", avgFG))
        print("  Apple MetalFX 空间超分:    \(metalFXStatus)")
        print(String(format: "  全链路单帧耗时 (去重+插帧+4K超分): %.2f ms / 16.67ms 预算", avgDedup + avgFG + metalFXMs))

        let dedupOK = (dupDropped >= (frameCount / 2) - 2) && (uniqueCount >= (frameCount / 2) - 2)
        let cutOK = (snap.sceneCutCount >= 1)
        let speedOK = (avgDedup + avgFG + metalFXMs < 16.67)
        print("  测试结果: dedup=\(dedupOK ? "PASS" : "FAIL"), sceneCut=\(cutOK ? "PASS" : "FAIL"), metalFX=\(metalFXOK ? "PASS" : "FAIL"), realtime60=\(speedOK ? "PASS" : "WARN")")
        print("============================================================")
        return dedupOK && cutOK && metalFXOK
    }

    private func getDroppedDupCount() -> Int { statsLock.lock(); defer { statsLock.unlock() }; return countDroppedDup }
    private func getUniqueCount() -> Int { statsLock.lock(); defer { statsLock.unlock() }; return countUnique }
    private func rawTelemetry() -> TelemetrySnapshot { statsLock.lock(); defer { statsLock.unlock() }; return _telemetry }

    // MARK: - Private GPU Helpers

    private func resolveVTKind(for cfg: PipelineConfig) -> VTSessionKind {
        switch cfg.frameGen {
        case .off, .metalOpticalFlow:
            return .none
        case .auto:
            if capabilities.supportsVTLowLatencyFG { return .lowLatencyTemporal2x }
            if capabilities.supportsVTFrameRateConv { return .frameRateConversion2x }
            return .none
        case .vtLowLatency2x:
            return capabilities.supportsVTLowLatencyFG ? .lowLatencyTemporal2x : .none
        case .vtLowLatency4x:
            return capabilities.supportsVTLowLatencyFG ? .lowLatencyTemporal4x : .none
        case .vtCombo4K60:
            return capabilities.supportsVTLowLatencyFG ? .lowLatencySpatial2xCombo : .none
        case .vtFrameRateConv:
            return capabilities.supportsVTFrameRateConv ? .frameRateConversion2x : .none
        }
    }

    private func ensureResources(width: Int, height: Int) {
        guard width != inputWidth || height != inputHeight || stagingBGRA == nil else { return }
        inputWidth = width
        inputHeight = height
        prevUniqueTex = nil
        prevUniqueVTBuffer = nil

        func makeTex(_ w: Int, _ h: Int, format: MTLPixelFormat, usage: MTLTextureUsage = [.shaderRead, .shaderWrite]) -> MTLTexture {
            let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: max(w, 1), height: max(h, 1), mipmapped: false)
            desc.storageMode = .private
            desc.usage = usage
            return device.makeTexture(descriptor: desc)!
        }

        stagingBGRA = makeTex(width, height, format: .bgra8Unorm)
        uniqueRing = (0..<4).map { _ in makeTex(width, height, format: .bgra8Unorm) }
        uniqueRingIndex = 0

        // 3 pyramid levels: W/2, W/4, W/8
        let sizes = [
            (max(width / 2, 16), max(height / 2, 16)),
            (max(width / 4, 8),  max(height / 4, 8)),
            (max(width / 8, 4),  max(height / 8, 4))
        ]
        lumaA  = sizes.map { makeTex($0.0, $0.1, format: .r16Float) }
        lumaB  = sizes.map { makeTex($0.0, $0.1, format: .r16Float) }
        flowAB = sizes.map { makeTex($0.0, $0.1, format: .rgba16Float) }
        flowBA = sizes.map { makeTex($0.0, $0.1, format: .rgba16Float) }
        flowSmoothAB = makeTex(sizes[0].0, sizes[0].1, format: .rgba16Float)
        flowSmoothBA = makeTex(sizes[0].0, sizes[0].1, format: .rgba16Float)

        // Double-buffered sets of 3 interpolated textures (supports up to 4x / 120Hz cleanly)
        ofOutputRing = (0..<2).map { _ in
            (0..<3).map { _ in makeTex(width, height, format: .bgra8Unorm) }
        }
        ofOutputBank = 0
    }

    @available(macOS 13.0, *)
    private func ensureMetalFXSpatialScaler(inW: Int, inH: Int, outW: Int, outH: Int) -> (MTLFXSpatialScaler, MTLTexture)? {
        if let existing = metalFXScalerBox as? MTLFXSpatialScaler,
           let tex = metalFXOutTex,
           metalFXInSize == (inW, inH),
           metalFXOutSize == (outW, outH) {
            return (existing, tex)
        }

        let desc = MTLFXSpatialScalerDescriptor()
        desc.inputWidth = inW
        desc.inputHeight = inH
        desc.outputWidth = outW
        desc.outputHeight = outH
        desc.colorTextureFormat = .bgra8Unorm
        desc.outputTextureFormat = .bgra8Unorm
        desc.colorProcessingMode = .perceptual

        guard let scaler = desc.makeSpatialScaler(device: device) else {
            return nil
        }

        let outDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: outW, height: outH, mipmapped: false)
        outDesc.storageMode = .private
        outDesc.usage = scaler.outputTextureUsage.union([.shaderRead, .renderTarget])
        guard let outTex = device.makeTexture(descriptor: outDesc) else { return nil }

        metalFXScalerBox = scaler
        metalFXOutTex = outTex
        metalFXInSize = (inW, inH)
        metalFXOutSize = (outW, outH)
        return (scaler, outTex)
    }

    private func encodeColorConvert(
        cmdBuf: MTLCommandBuffer,
        pixelBuffer: CVPixelBuffer,
        fourCC: OSType,
        colorRange: HDMIColorRange,
        outTex: MTLTexture
    ) {
        let w = outTex.width
        let h = outTex.height
        let fullFlag: UInt32 = (colorRange == .full || fourCC == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange) ? 1 : 0

        switch fourCC {
        case kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
             kCVPixelFormatType_420YpCbCr8BiPlanarFullRange:
            guard let yCV = wrapPlaneTexture(pixelBuffer, format: .r8Unorm, width: w, height: h, planeIndex: 0),
                  let uvCV = wrapPlaneTexture(pixelBuffer, format: .rg8Unorm, width: w / 2, height: h / 2, planeIndex: 1),
                  let yTex = CVMetalTextureGetTexture(yCV),
                  let uvTex = CVMetalTextureGetTexture(uvCV),
                  let enc = cmdBuf.makeComputeCommandEncoder() else { return }
            enc.setComputePipelineState(nv12ToBGRAPSO)
            enc.setTexture(yTex, index: 0)
            enc.setTexture(uvTex, index: 1)
            enc.setTexture(outTex, index: 2)
            var p = ColorConvertParams(width: UInt32(w), height: UInt32(h), isFullRange: fullFlag, isYUVS: 0)
            enc.setBytes(&p, length: MemoryLayout<ColorConvertParams>.stride, index: 0)
            dispatch2D(enc, width: w, height: h, pso: nv12ToBGRAPSO)
            enc.endEncoding()

        case kCVPixelFormatType_422YpCbCr8,
             kCVPixelFormatType_422YpCbCr8_yuvs:
            guard let packedCV = wrapPlaneTexture(pixelBuffer, format: .rgba8Unorm, width: w / 2, height: h, planeIndex: 0),
                  let packedTex = CVMetalTextureGetTexture(packedCV),
                  let enc = cmdBuf.makeComputeCommandEncoder() else { return }
            enc.setComputePipelineState(yuy2ToBGRAPSO)
            enc.setTexture(packedTex, index: 0)
            enc.setTexture(outTex, index: 1)
            let isYUVS: UInt32 = (fourCC == kCVPixelFormatType_422YpCbCr8_yuvs) ? 1 : 0
            var p = ColorConvertParams(width: UInt32(w), height: UInt32(h), isFullRange: fullFlag, isYUVS: isYUVS)
            enc.setBytes(&p, length: MemoryLayout<ColorConvertParams>.stride, index: 0)
            dispatch2D(enc, width: w / 2, height: h, pso: yuy2ToBGRAPSO)
            enc.endEncoding()

        default:
            guard let bgraCV = wrapPlaneTexture(pixelBuffer, format: .bgra8Unorm, width: w, height: h, planeIndex: 0),
                  let bgraTex = CVMetalTextureGetTexture(bgraCV),
                  let enc = cmdBuf.makeComputeCommandEncoder() else { return }
            enc.setComputePipelineState(copyBGRAPSO)
            enc.setTexture(bgraTex, index: 0)
            enc.setTexture(outTex, index: 1)
            dispatch2D(enc, width: w, height: h, pso: copyBGRAPSO)
            enc.endEncoding()
        }
    }

    private func encodeWriteToVTBuffer(
        cmdBuf: MTLCommandBuffer,
        srcBGRA: MTLTexture,
        dstVTBuffer: CVPixelBuffer,
        width: Int,
        height: Int
    ) {
        let fmt = CVPixelBufferGetPixelFormatType(dstVTBuffer)
        if fmt == kCVPixelFormatType_32BGRA {
            guard let cvTex = wrapPlaneTexture(dstVTBuffer, format: .bgra8Unorm, width: width, height: height, planeIndex: 0),
                  let dstTex = CVMetalTextureGetTexture(cvTex),
                  let enc = cmdBuf.makeComputeCommandEncoder() else { return }
            enc.setComputePipelineState(copyBGRAPSO)
            enc.setTexture(srcBGRA, index: 0)
            enc.setTexture(dstTex, index: 1)
            dispatch2D(enc, width: width, height: height, pso: copyBGRAPSO)
            enc.endEncoding()
        } else if fmt == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange || fmt == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange {
            guard let yCV = wrapPlaneTexture(dstVTBuffer, format: .r8Unorm, width: width, height: height, planeIndex: 0),
                  let uvCV = wrapPlaneTexture(dstVTBuffer, format: .rg8Unorm, width: width / 2, height: height / 2, planeIndex: 1),
                  let yTex = CVMetalTextureGetTexture(yCV),
                  let uvTex = CVMetalTextureGetTexture(uvCV),
                  let enc = cmdBuf.makeComputeCommandEncoder() else { return }
            enc.setComputePipelineState(bgraToNV12PSO)
            enc.setTexture(srcBGRA, index: 0)
            enc.setTexture(yTex, index: 1)
            enc.setTexture(uvTex, index: 2)
            dispatch2D(enc, width: width / 2, height: height / 2, pso: bgraToNV12PSO)
            enc.endEncoding()
        }
    }

    private func wrapVTOutputToPacedFrame(
        cmdBuf: MTLCommandBuffer,
        pb: CVPixelBuffer,
        pts: CMTime,
        isInterpolated: Bool
    ) -> PacedFrame? {
        let w = CVPixelBufferGetWidth(pb)
        let h = CVPixelBufferGetHeight(pb)
        let fmt = CVPixelBufferGetPixelFormatType(pb)

        if fmt == kCVPixelFormatType_32BGRA {
            guard let cvTex = wrapPlaneTexture(pb, format: .bgra8Unorm, width: w, height: h, planeIndex: 0),
                  let tex = CVMetalTextureGetTexture(cvTex) else { return nil }
            return PacedFrame(texture: tex, backingBuffer: pb, backingMetalTex: cvTex, pts: pts, isInterpolated: isInterpolated)
        } else {
            // Convert NV12 output from VT into a private BGRA texture for MetalFX / Present
            let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: w, height: h, mipmapped: false)
            desc.storageMode = .private
            desc.usage = [.shaderRead, .shaderWrite]
            guard let bgraOut = device.makeTexture(descriptor: desc) else { return nil }
            encodeColorConvert(cmdBuf: cmdBuf, pixelBuffer: pb, fourCC: fmt, colorRange: .limited, outTex: bgraOut)
            return PacedFrame(texture: bgraOut, backingBuffer: pb, pts: pts, isInterpolated: isInterpolated)
        }
    }

    /// Runs our custom 3-level pyramid bidirectional optical-flow + occlusion + HUD-guard frame generator.
    private func encodeMetalOpticalFlow(
        cmdBuf: MTLCommandBuffer,
        frameA: MTLTexture,
        frameB: MTLTexture,
        phases: [Float],
        hudProtection: Float
    ) -> [MTLTexture] {
        guard let smoothAB = flowSmoothAB, let smoothBA = flowSmoothBA else { return [] }

        // 1. Build 3-level luma pyramids (W/2 -> W/4 -> W/8) for A and B
        if let enc = cmdBuf.makeComputeCommandEncoder() {
            enc.setComputePipelineState(rgbToLumaHalfPSO)
            enc.setTexture(frameA, index: 0)
            enc.setTexture(lumaA[0], index: 1)
            dispatch2D(enc, width: lumaA[0].width, height: lumaA[0].height, pso: rgbToLumaHalfPSO)

            enc.setTexture(frameB, index: 0)
            enc.setTexture(lumaB[0], index: 1)
            dispatch2D(enc, width: lumaB[0].width, height: lumaB[0].height, pso: rgbToLumaHalfPSO)

            enc.setComputePipelineState(lumaDown2xPSO)
            for lvl in 1..<3 {
                enc.setTexture(lumaA[lvl - 1], index: 0)
                enc.setTexture(lumaA[lvl], index: 1)
                dispatch2D(enc, width: lumaA[lvl].width, height: lumaA[lvl].height, pso: lumaDown2xPSO)

                enc.setTexture(lumaB[lvl - 1], index: 0)
                enc.setTexture(lumaB[lvl], index: 1)
                dispatch2D(enc, width: lumaB[lvl].width, height: lumaB[lvl].height, pso: lumaDown2xPSO)
            }

            // 2. Coarse optical flow at level 2 (W/8 x H/8) in both directions (A->B and B->A)
            enc.setComputePipelineState(flowCoarsePSO)
            enc.setTexture(lumaA[2], index: 0)
            enc.setTexture(lumaB[2], index: 1)
            enc.setTexture(flowAB[2], index: 2)
            dispatch2D(enc, width: flowAB[2].width, height: flowAB[2].height, pso: flowCoarsePSO)

            enc.setTexture(lumaB[2], index: 0)
            enc.setTexture(lumaA[2], index: 1)
            enc.setTexture(flowBA[2], index: 2)
            dispatch2D(enc, width: flowBA[2].width, height: flowBA[2].height, pso: flowCoarsePSO)

            // 3. Hierarchical refinement: level 2 -> level 1 -> level 0 (W/2 x H/2)
            enc.setComputePipelineState(flowRefinePSO)
            for lvl in [1, 0] {
                enc.setTexture(lumaA[lvl], index: 0)
                enc.setTexture(lumaB[lvl], index: 1)
                enc.setTexture(flowAB[lvl + 1], index: 2)
                enc.setTexture(flowAB[lvl], index: 3)
                dispatch2D(enc, width: flowAB[lvl].width, height: flowAB[lvl].height, pso: flowRefinePSO)

                enc.setTexture(lumaB[lvl], index: 0)
                enc.setTexture(lumaA[lvl], index: 1)
                enc.setTexture(flowBA[lvl + 1], index: 2)
                enc.setTexture(flowBA[lvl], index: 3)
                dispatch2D(enc, width: flowBA[lvl].width, height: flowBA[lvl].height, pso: flowRefinePSO)
            }

            // 4. Edge-aware bilateral smoothing on final half-res flow
            enc.setComputePipelineState(flowSmoothPSO)
            enc.setTexture(flowAB[0], index: 0)
            enc.setTexture(lumaA[0], index: 1)
            enc.setTexture(smoothAB, index: 2)
            dispatch2D(enc, width: smoothAB.width, height: smoothAB.height, pso: flowSmoothPSO)

            enc.setTexture(flowBA[0], index: 0)
            enc.setTexture(lumaB[0], index: 1)
            enc.setTexture(smoothBA, index: 2)
            dispatch2D(enc, width: smoothBA.width, height: smoothBA.height, pso: flowSmoothPSO)

            // 5. Bidirectional warp + occlusion handling + static HUD guard for each requested phase
            let bank = ofOutputRing[ofOutputBank]
            ofOutputBank = (ofOutputBank + 1) % ofOutputRing.count

            enc.setComputePipelineState(flowWarpPSO)
            var outputs: [MTLTexture] = []
            for (idx, phase) in phases.enumerated() where idx < bank.count {
                let outTex = bank[idx]
                enc.setTexture(frameA, index: 0)
                enc.setTexture(frameB, index: 1)
                enc.setTexture(smoothAB, index: 2)
                enc.setTexture(smoothBA, index: 3)
                enc.setTexture(outTex, index: 4)
                var wp = WarpParams(
                    phase: phase,
                    hudProtection: hudProtection,
                    flowScaleX: 1.0 / Float(smoothAB.width),
                    flowScaleY: 1.0 / Float(smoothAB.height)
                )
                enc.setBytes(&wp, length: MemoryLayout<WarpParams>.stride, index: 0)
                dispatch2D(enc, width: outTex.width, height: outTex.height, pso: flowWarpPSO)
                outputs.append(outTex)
            }
            enc.endEncoding()
            return outputs
        }
        return []
    }

    private func wrapPlaneTexture(
        _ pixelBuffer: CVPixelBuffer,
        format: MTLPixelFormat,
        width: Int,
        height: Int,
        planeIndex: Int
    ) -> CVMetalTexture? {
        guard let cache = textureCache else { return nil }
        var cvTex: CVMetalTexture?
        let status = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault,
            cache,
            pixelBuffer,
            nil,
            format,
            width,
            height,
            planeIndex,
            &cvTex
        )
        return status == kCVReturnSuccess ? cvTex : nil
    }

    private func dispatch2D(_ enc: MTLComputeCommandEncoder, width: Int, height: Int, pso: MTLComputePipelineState) {
        let tw = min(16, pso.threadExecutionWidth)
        let th = max(1, min(16, pso.maxTotalThreadsPerThreadgroup / tw))
        let tg = MTLSize(width: tw, height: th, depth: 1)
        let grid = MTLSize(width: (width + tw - 1) / tw, height: (height + th - 1) / th, depth: 1)
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
    }

    private func aspectFittedSize(srcW: Int, srcH: Int, dstW: Int, dstH: Int) -> (Int, Int) {
        guard srcW > 0, srcH > 0, dstW > 0, dstH > 0 else { return (dstW, dstH) }
        let srcAspect = Double(srcW) / Double(srcH)
        let dstAspect = Double(dstW) / Double(dstH)
        if dstAspect > srcAspect {
            let fitH = dstH
            let fitW = max(1, Int(round(Double(fitH) * srcAspect)))
            return (fitW, fitH)
        } else {
            let fitW = dstW
            let fitH = max(1, Int(round(Double(fitW) / srcAspect)))
            return (fitW, fitH)
        }
    }
}
