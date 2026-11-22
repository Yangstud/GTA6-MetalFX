import CoreMedia
import CoreVideo
import Foundation
import Metal
import MetalFX
import VideoToolbox

/// Describes the capabilities of the host Mac's Apple VideoToolbox & MetalFX engines.
public struct AppleHardwareCapabilities {
    public let deviceName: String
    public let supportsMetalFXSpatial: Bool
    public let supportsVTLowLatencyFG: Bool
    public let supportsVTLowLatencySR: Bool
    public let supportsVTFrameRateConv: Bool

    public var summaryString: String {
        var tags: [String] = []
        if supportsVTLowLatencyFG { tags.append("VT-AI-FG(macOS26)") }
        else if supportsVTFrameRateConv { tags.append("VT-FRC(15.4+)") }
        else { tags.append("Metal-OF-FG") }

        if supportsMetalFXSpatial { tags.append("MetalFX-SR") }
        if supportsVTLowLatencySR { tags.append("VT-AI-SR") }
        if !supportsMetalFXSpatial && !supportsVTLowLatencySR { tags.append("CAS-Upscale") }
        return "\(deviceName) | " + tags.joined(separator: " + ")
    }
}

/// Active mode inside `VTFrameGenEngine`.
public enum VTSessionKind: Equatable {
    case none
    case lowLatencyTemporal2x       // macOS 26+: 30 -> 60 (1 interp @ 0.5)
    case lowLatencyTemporal4x       // macOS 26+: 30 -> 120 (3 interps @ 0.25, 0.5, 0.75)
    case lowLatencySpatial2xCombo   // macOS 26+: 1080p30 -> 4K60 joint AI SR + AI FG!
    case frameRateConversion2x      // macOS 15.4+: optical-flow ML FRC (1 interp @ 0.5)
}

/// Result produced by a single VideoToolbox AI Frame Generation invocation.
public struct VTInterpolatedOutput {
    /// Chronologically ordered interpolated intermediate frames (e.g. [0.5] or [0.25, 0.5, 0.75]).
    public let intermediateBuffers: [CVPixelBuffer]
    /// When `lowLatencySpatial2xCombo` is active, Apple's ML model ALSO returns the 2x AI-upscaled source frame!
    public let upscaledSourceBuffer: CVPixelBuffer?
}

/// Zero-copy, asynchronously warmed wrapper around Apple's official macOS 26 `VTLowLatencyFrameInterpolation`
/// / `VTLowLatencySuperResolutionScaler` and macOS 15.4 `VTFrameRateConversion` APIs.
///
/// Design invariants:
/// 1. Never blocks the UI/render thread during ML model warmup (`startSession`).
/// 2. Resolves `CVPixelBufferPool` attributes against Apple's `sourcePixelBufferAttributes` /
///    `destinationPixelBufferAttributes` + `kCVPixelBufferMetalCompatibilityKey: true` so Metal & ANE
///    share the exact same IOSurface with zero CPU memcpy.
/// 3. Falls back cleanly to the Metal Pyramid Optical Flow engine if hardware or OS doesn't support VT ML.
public final class VTFrameGenEngine {
    public private(set) var activeKind: VTSessionKind = .none
    public private(set) var isReady: Bool = false
    public private(set) var lastError: String? = nil
    public private(set) var sourcePixelFormat: OSType = kCVPixelFormatType_32BGRA

    private let lock = NSLock()
    private let warmupQueue = DispatchQueue(label: "com.cm716fx.vt.warmup", qos: .userInitiated)

    // Stored as `AnyObject` so the class compiles cleanly with deployment target macOS 14.0.
    private var vtProcessorBox: AnyObject?
    private var vtSRProcessorBox: AnyObject?
    private var vtSRReady: Bool = false
    private var vtSRScaleFactor: Float = 2.0

    private var configuredWidth: Int = 0
    private var configuredHeight: Int = 0
    private var sourcePool: CVPixelBufferPool?
    private var destPool: CVPixelBufferPool?
    private var srDestPool: CVPixelBufferPool?

    public init() {}

    public static func isVTFrameRateConvSupported() -> Bool {
        if #available(macOS 26.0, *) {
            return VTFrameRateConversionConfiguration.isSupported
        } else if #available(macOS 15.4, *) {
            if VTFrameRateConversionConfiguration.responds(to: Selector(("isSupported"))) {
                return VTFrameRateConversionConfiguration.isSupported
            }
            return VTFrameRateConversionConfiguration.processorSupported
        }
        return false
    }

    public static func probeCapabilities(device: MTLDevice) -> AppleHardwareCapabilities {
        var supportsMetalFX = false
        var supportsLowLatencyFG = false
        var supportsLowLatencySR = false
        var supportsFRC = false

        if #available(macOS 13.0, *) {
            supportsMetalFX = MTLFXSpatialScalerDescriptor.supportsDevice(device)
        }
        if #available(macOS 26.0, *) {
            supportsLowLatencyFG = VTLowLatencyFrameInterpolationConfiguration.isSupported
            supportsLowLatencySR = VTLowLatencySuperResolutionScalerConfiguration.isSupported
        }
        supportsFRC = isVTFrameRateConvSupported()

        return AppleHardwareCapabilities(
            deviceName: device.name,
            supportsMetalFXSpatial: supportsMetalFX,
            supportsVTLowLatencyFG: supportsLowLatencyFG,
            supportsVTLowLatencySR: supportsLowLatencySR,
            supportsVTFrameRateConv: supportsFRC
        )
    }

    /// Asynchronously prepares a `VTFrameProcessor` session for the requested resolution & mode.
    public func prepareSessionAsync(kind: VTSessionKind, width: Int, height: Int) {
        lock.lock()
        let alreadyMatches = (activeKind == kind && configuredWidth == width && configuredHeight == height && (isReady || vtProcessorBox != nil))
        lock.unlock()
        if alreadyMatches { return }

        if kind == .none {
            teardownSession()
            return
        }

        warmupQueue.async { [weak self] in
            guard let self else { return }
            self.buildSession(kind: kind, width: width, height: height)
        }
    }

    /// Asynchronously prepares standalone `VTLowLatencySuperResolutionScaler` (macOS 26+).
    public func prepareSuperResAsync(width: Int, height: Int) {
        guard #available(macOS 26.0, *),
              VTLowLatencySuperResolutionScalerConfiguration.isSupported else {
            return
        }
        warmupQueue.async { [weak self] in
            guard let self else { return }
            self.buildSuperResSession(width: width, height: height)
        }
    }

    public func teardownSession() {
        lock.lock()
        let oldProc = vtProcessorBox
        vtProcessorBox = nil
        activeKind = .none
        isReady = false
        sourcePool = nil
        destPool = nil
        lock.unlock()

        if #available(macOS 15.4, *), let proc = oldProc as? VTFrameProcessor {
            proc.endSession()
        }
    }

    /// Allocates an IOSurface-backed, Metal-compatible source pixel buffer guaranteed accepted by VTFrameProcessor.
    public func allocateSourceBuffer() -> CVPixelBuffer? {
        lock.lock()
        let pool = sourcePool
        lock.unlock()
        guard let pool else { return nil }
        var pb: CVPixelBuffer?
        let status = CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &pb)
        return status == kCVReturnSuccess ? pb : nil
    }

    private func allocateDestBuffer() -> CVPixelBuffer? {
        lock.lock()
        let pool = destPool
        lock.unlock()
        guard let pool else { return nil }
        var pb: CVPixelBuffer?
        let status = CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &pb)
        return status == kCVReturnSuccess ? pb : nil
    }

    /// Encodes AI frame interpolation directly into the supplied Metal command buffer (zero CPU stall!).
    /// Returns `nil` if the VT ML session is still warming up or unsupported on this Mac.
    public func encodeInterpolation(
        commandBuffer: MTLCommandBuffer,
        previousBuffer: CVPixelBuffer,
        previousPTS: CMTime,
        currentBuffer: CVPixelBuffer,
        currentPTS: CMTime,
        isSceneCut: Bool
    ) -> VTInterpolatedOutput? {
        lock.lock()
        let ready = isReady
        let kind = activeKind
        let procObj = vtProcessorBox
        lock.unlock()

        guard ready, let procObj else { return nil }

        if #available(macOS 26.0, *), let processor = procObj as? VTFrameProcessor {
            switch kind {
            case .lowLatencyTemporal2x:
                guard let outMid = allocateDestBuffer(),
                      let prevFrame = VTFrameProcessorFrame(buffer: previousBuffer, presentationTimeStamp: previousPTS),
                      let currFrame = VTFrameProcessorFrame(buffer: currentBuffer, presentationTimeStamp: currentPTS) else {
                    return nil
                }
                let midPTS = midpointTime(previousPTS, currentPTS, phase: 0.5)
                guard let dstFrame = VTFrameProcessorFrame(buffer: outMid, presentationTimeStamp: midPTS),
                      let params = VTLowLatencyFrameInterpolationParameters(
                          sourceFrame: currFrame,
                          previousFrame: prevFrame,
                          interpolationPhase: [0.5],
                          destinationFrames: [dstFrame]
                      ) else {
                    return nil
                }
                processor.process(with: commandBuffer, parameters: params)
                return VTInterpolatedOutput(intermediateBuffers: [outMid], upscaledSourceBuffer: nil)

            case .lowLatencyTemporal4x:
                guard let b1 = allocateDestBuffer(),
                      let b2 = allocateDestBuffer(),
                      let b3 = allocateDestBuffer(),
                      let prevFrame = VTFrameProcessorFrame(buffer: previousBuffer, presentationTimeStamp: previousPTS),
                      let currFrame = VTFrameProcessorFrame(buffer: currentBuffer, presentationTimeStamp: currentPTS) else {
                    return nil
                }
                let phases: [Float] = [0.25, 0.50, 0.75]
                let bufs = [b1, b2, b3]
                var dstFrames: [VTFrameProcessorFrame] = []
                for (pb, ph) in zip(bufs, phases) {
                    guard let f = VTFrameProcessorFrame(buffer: pb, presentationTimeStamp: midpointTime(previousPTS, currentPTS, phase: Double(ph))) else {
                        return nil
                    }
                    dstFrames.append(f)
                }
                guard let params = VTLowLatencyFrameInterpolationParameters(
                    sourceFrame: currFrame,
                    previousFrame: prevFrame,
                    interpolationPhase: phases,
                    destinationFrames: dstFrames
                ) else {
                    return nil
                }
                processor.process(with: commandBuffer, parameters: params)
                return VTInterpolatedOutput(intermediateBuffers: bufs, upscaledSourceBuffer: nil)

            case .lowLatencySpatial2xCombo:
                // Per Apple header: spatialScaleFactor=2 outputs BOTH upscaled intermediate (0.5) AND upscaled sourceFrame!
                guard let outMid4K = allocateDestBuffer(),
                      let outSrc4K = allocateDestBuffer(),
                      let prevFrame = VTFrameProcessorFrame(buffer: previousBuffer, presentationTimeStamp: previousPTS),
                      let currFrame = VTFrameProcessorFrame(buffer: currentBuffer, presentationTimeStamp: currentPTS) else {
                    return nil
                }
                let midPTS = midpointTime(previousPTS, currentPTS, phase: 0.5)
                guard let dstMid = VTFrameProcessorFrame(buffer: outMid4K, presentationTimeStamp: midPTS),
                      let dstSrc = VTFrameProcessorFrame(buffer: outSrc4K, presentationTimeStamp: currentPTS),
                      let params = VTLowLatencyFrameInterpolationParameters(
                          sourceFrame: currFrame,
                          previousFrame: prevFrame,
                          interpolationPhase: [0.5],
                          destinationFrames: [dstMid, dstSrc]
                      ) else {
                    return nil
                }
                processor.process(with: commandBuffer, parameters: params)
                return VTInterpolatedOutput(intermediateBuffers: [outMid4K], upscaledSourceBuffer: outSrc4K)

            case .frameRateConversion2x:
                break // handled below
            case .none:
                return nil
            }
        }

        // macOS 15.4+ fallback: VTFrameRateConversion
        if #available(macOS 15.4, *),
           kind == .frameRateConversion2x,
           let processor = procObj as? VTFrameProcessor {
            guard let outMid = allocateDestBuffer(),
                  let prevFrame = VTFrameProcessorFrame(buffer: previousBuffer, presentationTimeStamp: previousPTS),
                  let nextFrame = VTFrameProcessorFrame(buffer: currentBuffer, presentationTimeStamp: currentPTS) else {
                return nil
            }
            let midPTS = midpointTime(previousPTS, currentPTS, phase: 0.5)
            guard let dstFrame = VTFrameProcessorFrame(buffer: outMid, presentationTimeStamp: midPTS),
                  let params = VTFrameRateConversionParameters(
                      sourceFrame: prevFrame,
                      nextFrame: nextFrame,
                      opticalFlow: nil,
                      interpolationPhase: [0.5],
                      submissionMode: isSceneCut ? .random : .sequential,
                      destinationFrames: [dstFrame]
                  ) else {
                return nil
            }
            processor.process(with: commandBuffer, parameters: params)
            return VTInterpolatedOutput(intermediateBuffers: [outMid], upscaledSourceBuffer: nil)
        }

        return nil
    }

    /// Encodes standalone Apple AI Low-Latency Super Resolution (macOS 26+) directly into the Metal command buffer.
    public func encodeSuperResolution(
        commandBuffer: MTLCommandBuffer,
        sourceBuffer: CVPixelBuffer,
        pts: CMTime
    ) -> CVPixelBuffer? {
        guard #available(macOS 26.0, *) else { return nil }
        lock.lock()
        let ready = vtSRReady
        let procObj = vtSRProcessorBox
        let pool = srDestPool
        lock.unlock()

        guard ready, let processor = procObj as? VTFrameProcessor, let pool else { return nil }
        var dstPB: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &dstPB) == kCVReturnSuccess,
              let outPB = dstPB,
              let srcFrame = VTFrameProcessorFrame(buffer: sourceBuffer, presentationTimeStamp: pts),
              let dstFrame = VTFrameProcessorFrame(buffer: outPB, presentationTimeStamp: pts) else {
            return nil
        }
        let params = VTLowLatencySuperResolutionScalerParameters(sourceFrame: srcFrame, destinationFrame: dstFrame)
        processor.process(with: commandBuffer, parameters: params)
        return outPB
    }

    // MARK: - Private Session Builders

    private func buildSession(kind: VTSessionKind, width: Int, height: Int) {
        teardownSession()

        if #available(macOS 26.0, *) {
            var config: VTLowLatencyFrameInterpolationConfiguration?
            switch kind {
            case .lowLatencyTemporal2x:
                if VTLowLatencyFrameInterpolationConfiguration.isSupported {
                    config = VTLowLatencyFrameInterpolationConfiguration(
                        frameWidth: width,
                        frameHeight: height,
                        numberOfInterpolatedFrames: 1
                    )
                }
            case .lowLatencyTemporal4x:
                if VTLowLatencyFrameInterpolationConfiguration.isSupported {
                    config = VTLowLatencyFrameInterpolationConfiguration(
                        frameWidth: width,
                        frameHeight: height,
                        numberOfInterpolatedFrames: 2
                    )
                }
            case .lowLatencySpatial2xCombo:
                if VTLowLatencyFrameInterpolationConfiguration.isSupported {
                    config = VTLowLatencyFrameInterpolationConfiguration(
                        frameWidth: width,
                        frameHeight: height,
                        spatialScaleFactor: 2
                    )
                }
            default:
                break
            }

            if let cfg = config {
                let proc = VTFrameProcessor()
                do {
                    try proc.startSession(configuration: cfg)
                    let preferredFmt = selectPreferredPixelFormat(cfg.supportedPixelFormats)
                    let outW = width * max(cfg.spatialScaleFactor, 1)
                    let outH = height * max(cfg.spatialScaleFactor, 1)
                    let srcPool = createResolvedPool(
                        baseAttributes: cfg.sourcePixelBufferAttributes,
                        width: width,
                        height: height,
                        pixelFormat: preferredFmt
                    )
                    let dstPool = createResolvedPool(
                        baseAttributes: cfg.destinationPixelBufferAttributes,
                        width: outW,
                        height: outH,
                        pixelFormat: preferredFmt
                    )
                    lock.lock()
                    self.vtProcessorBox = proc
                    self.activeKind = kind
                    self.configuredWidth = width
                    self.configuredHeight = height
                    self.sourcePixelFormat = preferredFmt
                    self.sourcePool = srcPool
                    self.destPool = dstPool
                    self.isReady = (srcPool != nil && dstPool != nil)
                    self.lastError = nil
                    lock.unlock()
                    return
                } catch {
                    lock.lock()
                    self.lastError = "VT LowLatencyFG 启动失败: \(error.localizedDescription)"
                    lock.unlock()
                }
            }
        }

        // Fallback: macOS 15.4+ VTFrameRateConversion
        if #available(macOS 15.4, *),
           kind == .frameRateConversion2x,
           Self.isVTFrameRateConvSupported(),
           let cfg = VTFrameRateConversionConfiguration(
               frameWidth: width,
               frameHeight: height,
               usePrecomputedFlow: false,
               qualityPrioritization: .normal,
               revision: .revision1
           ) {
            let proc = VTFrameProcessor()
            do {
                try proc.startSession(configuration: cfg)
                // Avoid macOS 26-only Swift overlay `supportedPixelFormats` symbol on macOS 15.x!
                let supportedList: [OSType]
                if #available(macOS 26.0, *) {
                    supportedList = cfg.supportedPixelFormats
                } else {
                    supportedList = cfg.frameSupportedPixelFormats.map { OSType($0.uint32Value) }
                }
                let preferredFmt = selectPreferredPixelFormat(supportedList)
                let srcPool = createResolvedPool(
                    baseAttributes: cfg.sourcePixelBufferAttributes,
                    width: width,
                    height: height,
                    pixelFormat: preferredFmt
                )
                let dstPool = createResolvedPool(
                    baseAttributes: cfg.destinationPixelBufferAttributes,
                    width: width,
                    height: height,
                    pixelFormat: preferredFmt
                )
                lock.lock()
                self.vtProcessorBox = proc
                self.activeKind = .frameRateConversion2x
                self.configuredWidth = width
                self.configuredHeight = height
                self.sourcePixelFormat = preferredFmt
                self.sourcePool = srcPool
                self.destPool = dstPool
                self.isReady = (srcPool != nil && dstPool != nil)
                self.lastError = nil
                lock.unlock()
            } catch {
                lock.lock()
                self.lastError = "VT FRC 启动失败: \(error.localizedDescription)"
                lock.unlock()
            }
        }
    }

    @available(macOS 26.0, *)
    private func buildSuperResSession(width: Int, height: Int) {
        let factors = VTLowLatencySuperResolutionScalerConfiguration.supportedScaleFactors(
            frameWidth: width,
            frameHeight: height
        )
        guard let scale = factors.contains(2.0) ? Float(2.0) : factors.first else { return }
        let cfg = VTLowLatencySuperResolutionScalerConfiguration(
            frameWidth: width,
            frameHeight: height,
            scaleFactor: scale
        )
        let proc = VTFrameProcessor()
        do {
            try proc.startSession(configuration: cfg)
            let preferredFmt = selectPreferredPixelFormat(cfg.supportedPixelFormats)
            let outW = Int(round(Float(width) * scale))
            let outH = Int(round(Float(height) * scale))
            let pool = createResolvedPool(
                baseAttributes: cfg.destinationPixelBufferAttributes,
                width: outW,
                height: outH,
                pixelFormat: preferredFmt
            )
            lock.lock()
            self.vtSRProcessorBox = proc
            self.vtSRScaleFactor = scale
            self.srDestPool = pool
            self.vtSRReady = (pool != nil)
            lock.unlock()
        } catch {
            lock.lock()
            self.vtSRReady = false
            lock.unlock()
        }
    }

    private func selectPreferredPixelFormat(_ supported: [OSType]) -> OSType {
        // Prefer BGRA for direct Metal compute interop without extra chroma plane writes.
        if supported.contains(kCVPixelFormatType_32BGRA) {
            return kCVPixelFormatType_32BGRA
        }
        return supported.first ?? kCVPixelFormatType_32BGRA
    }

    private func createResolvedPool(
        baseAttributes: [String: Any],
        width: Int,
        height: Int,
        pixelFormat: OSType
    ) -> CVPixelBufferPool? {
        let metalAttrs: [String: Any] = [
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferPixelFormatTypeKey as String: pixelFormat,
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any]
        ]

        var resolved: CFDictionary?
        let list = [baseAttributes as CFDictionary, metalAttrs as CFDictionary] as CFArray
        let resolveStatus = CVPixelBufferCreateResolvedAttributesDictionary(kCFAllocatorDefault, list, &resolved)
        let finalAttrs: CFDictionary = (resolveStatus == kCVReturnSuccess && resolved != nil)
            ? resolved!
            : (metalAttrs as CFDictionary)

        let poolAttrs: [String: Any] = [
            kCVPixelBufferPoolMinimumBufferCountKey as String: 6
        ]
        var pool: CVPixelBufferPool?
        let status = CVPixelBufferPoolCreate(kCFAllocatorDefault, poolAttrs as CFDictionary, finalAttrs, &pool)
        return status == kCVReturnSuccess ? pool : nil
    }

    private func midpointTime(_ a: CMTime, _ b: CMTime, phase: Double) -> CMTime {
        let diff = CMTimeSubtract(b, a)
        if !diff.isValid || diff.seconds <= 0 {
            return CMTimeAdd(a, CMTime(seconds: 0.016667 * phase, preferredTimescale: 60000))
        }
        return CMTimeAdd(a, CMTime(seconds: diff.seconds * phase, preferredTimescale: 60000))
    }
}
