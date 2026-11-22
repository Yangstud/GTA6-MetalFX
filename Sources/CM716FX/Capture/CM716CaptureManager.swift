import AVFoundation
import CoreMedia
import CoreVideo
import Foundation

/// Represents a discovered video capture device with CM716/MS2130 scoring.
public struct CaptureDeviceInfo: Equatable {
    public let uniqueID: String
    public let localizedName: String
    public let manufacturer: String
    public let isLikelyCM716: Bool
}

/// Callback delegate receiving raw zero-copy CVPixelBuffers from CM716.
public protocol CM716CaptureDelegate: AnyObject {
    func captureManager(
        _ manager: CM716CaptureManager,
        didOutput pixelBuffer: CVPixelBuffer,
        presentationTime: CMTime,
        fourCC: OSType
    )
    func captureManager(
        _ manager: CM716CaptureManager,
        didUpdateStatus deviceName: String,
        fourCC: String,
        resolution: String,
        warning: String?
    )
}

/// Dedicated UVC/UAC capture manager tuned specifically for UGREEN CM716 (MacroSilicon MS2130).
/// Avoids MJPEG bandwidth/latency traps and locks onto uncompressed NV12/YUY2 IOSurface buffers.
public final class CM716CaptureManager: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    public weak var delegate: CM716CaptureDelegate?

    private let session = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "com.cm716fx.capture.session", qos: .userInteractive)
    private let videoQueue = DispatchQueue(label: "com.cm716fx.capture.video", qos: .userInteractive)

    private var videoInput: AVCaptureDeviceInput?
    private var videoOutput: AVCaptureVideoDataOutput?
    private var audioInput: AVCaptureDeviceInput?
    private var audioPreviewOutput: AVCaptureAudioPreviewOutput?

    private var activeDeviceID: String?
    private var activeProfile: CaptureProfile = .hz60NV12Dedup
    private var isAudioMuted: Bool = false
    private var audioVolume: Float = 1.0

    public private(set) var isRunning: Bool = false

    public override init() {
        super.init()
    }

    /// Discovers all connected video capture devices, ranking UGREEN CM716 / MS2130 / USB3 Capture first.
    public func listVideoDevices() -> [CaptureDeviceInfo] {
        var deviceTypes: [AVCaptureDevice.DeviceType] = [.external, .builtInWideAngleCamera]
        if #available(macOS 14.0, *) {
            deviceTypes = [.external, .continuityCamera, .builtInWideAngleCamera]
        }
        let discovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: deviceTypes,
            mediaType: .video,
            position: .unspecified
        )
        let mapped = discovery.devices.map { dev -> CaptureDeviceInfo in
            let lower = "\(dev.localizedName) \(dev.manufacturer)".lowercased()
            let isVirtualOrPhone = lower.contains("iphone")
                || lower.contains("ipad")
                || lower.contains("desk view")
                || lower.contains("obs virtual")
                || lower.contains("continuity")
            let isExplicitCapture = lower.contains("cm716")
                || lower.contains("25854")
                || lower.contains("ugreen")
                || lower.contains("ms2130")
                || lower.contains("ms2131")
                || lower.contains("macrosilicon")
                || lower.contains("usb3")
                || lower.contains("capture")
                || lower.contains("hdmi")
                || lower.contains("video grabber")
                || lower.contains("elgato")
                || lower.contains("avermedia")
            let likely = isExplicitCapture || (dev.deviceType == .external && !isVirtualOrPhone)
            return CaptureDeviceInfo(
                uniqueID: dev.uniqueID,
                localizedName: dev.localizedName,
                manufacturer: dev.manufacturer,
                isLikelyCM716: likely
            )
        }
        return mapped.sorted { lhs, rhs in
            if lhs.isLikelyCM716 != rhs.isLikelyCM716 { return lhs.isLikelyCM716 }
            return lhs.localizedName < rhs.localizedName
        }
    }

    /// Starts capture on the preferred CM716 device (or explicit `deviceID`).
    public func start(deviceID: String? = nil, profile: CaptureProfile) {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            self.activeProfile = profile
            self.configureAndStartLocked(preferredDeviceID: deviceID)
        }
    }

    public func stop() {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            if self.session.isRunning {
                self.session.stopRunning()
            }
            self.isRunning = false
        }
    }

    public func setProfile(_ profile: CaptureProfile) {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            guard self.activeProfile != profile else { return }
            self.activeProfile = profile
            if self.isRunning || self.activeDeviceID != nil {
                self.configureAndStartLocked(preferredDeviceID: self.activeDeviceID)
            }
        }
    }

    public func setAudio(muted: Bool, volume: Float) {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            self.isAudioMuted = muted
            self.audioVolume = volume
            self.audioPreviewOutput?.volume = muted ? 0.0 : volume
        }
    }

    private func configureAndStartLocked(preferredDeviceID: String?) {
        let devices = listVideoDevices()
        let chosenInfo: CaptureDeviceInfo?
        if let id = preferredDeviceID {
            chosenInfo = devices.first(where: { $0.uniqueID == id })
        } else {
            // Auto-pick external capture card only; do NOT silently grab FaceTime webcam unless user picks it.
            chosenInfo = devices.first(where: { $0.isLikelyCM716 })
        }

        guard let targetInfo = chosenInfo,
              let device = AVCaptureDevice(uniqueID: targetInfo.uniqueID) else {
            if session.isRunning { session.stopRunning() }
            isRunning = false
            delegate?.captureManager(
                self,
                didUpdateStatus: "未检测到 CM716 / 外置采集卡 (按 T 开启内置测试画面)",
                fourCC: "----",
                resolution: "0x0",
                warning: nil
            )
            return
        }

        activeDeviceID = device.uniqueID
        session.beginConfiguration()

        // Remove previous inputs/outputs cleanly
        if let oldIn = videoInput { session.removeInput(oldIn); videoInput = nil }
        if let oldOut = videoOutput { session.removeOutput(oldOut); videoOutput = nil }
        if let oldAudIn = audioInput { session.removeInput(oldAudIn); audioInput = nil }
        if let oldAudOut = audioPreviewOutput { session.removeOutput(oldAudOut); audioPreviewOutput = nil }

        // 1. Add video input
        guard let newVideoInput = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(newVideoInput) else {
            session.commitConfiguration()
            delegate?.captureManager(
                self,
                didUpdateStatus: "\(device.localizedName) (无相机权限，请在系统设置开启)",
                fourCC: "ERR!",
                resolution: "0x0",
                warning: "请在「系统设置 → 隐私与安全性 → 摄像头」允许 CM716FX 访问采集卡"
            )
            return
        }
        session.addInput(newVideoInput)
        videoInput = newVideoInput

        // 2. Negotiate optimal CM716 (MS2130) format & FPS
        let negotiated = negotiateOptimalFormat(for: device, profile: activeProfile)
        var warningMsg: String? = nil
        var activeFourCCStr = "AUTO"
        var activeResStr = "1920x1080"

        if let best = negotiated {
            do {
                try device.lockForConfiguration()
                device.activeFormat = best.format
                // MUST use exact CMTime durations from videoSupportedFrameRateRanges (e.g. MS2130 uses 1000000/60000240, not 1/60!)
                if best.minFrameDuration.isValid && best.maxFrameDuration.isValid {
                    device.activeVideoMinFrameDuration = best.minFrameDuration
                    device.activeVideoMaxFrameDuration = best.maxFrameDuration
                }
                device.unlockForConfiguration()

                let dims = CMVideoFormatDescriptionGetDimensions(best.format.formatDescription)
                let subType = CMFormatDescriptionGetMediaSubType(best.format.formatDescription)
                activeFourCCStr = fourCCString(subType)
                activeResStr = "\(dims.width)x\(dims.height)@\(best.fps)"

                if subType == kCMVideoCodecType_JPEG_OpenDML || activeFourCCStr.lowercased().contains("dmb1") || activeFourCCStr.lowercased().contains("mjpg") {
                    warningMsg = "⚠️ 当前协商为 MJPEG（疑似插在 USB 2.0 口/转接头），请直插 Mac 的 USB-C/USB 3.0 口以解锁无损 NV12/YUY2！"
                }
            } catch {
                warningMsg = "锁定采集格式失败: \(error.localizedDescription)"
            }
        }

        // 3. Configure zero-copy IOSurface + Metal compatible output
        let output = AVCaptureVideoDataOutput()
        output.alwaysDiscardsLateVideoFrames = true

        // Keep native uncompressed YUV format from CM716 when possible; let GPU do BT.709 conversion zero-copy.
        let desiredOutputFormat: OSType
        if let best = negotiated {
            let subType = CMFormatDescriptionGetMediaSubType(best.format.formatDescription)
            switch subType {
            case kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
                 kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
                 kCVPixelFormatType_422YpCbCr8,
                 kCVPixelFormatType_422YpCbCr8_yuvs,
                 kCVPixelFormatType_32BGRA:
                desiredOutputFormat = subType
            default:
                // If hardware only offered MJPEG (USB 2.0 fallback), ask CoreMedia to hardware-decode into NV12.
                desiredOutputFormat = kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
            }
        } else {
            desiredOutputFormat = kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
        }

        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: desiredOutputFormat,
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any]
        ]
        output.setSampleBufferDelegate(self, queue: videoQueue)

        if session.canAddOutput(output) {
            session.addOutput(output)
            videoOutput = output
        }

        // 4. Auto-bind CM716 UAC 24-bit/48kHz HDMI Audio Passthrough (ultra-low latency)
        attachMatchingAudioLocked(videoDeviceName: device.localizedName)

        // MUST commit configuration BEFORE calling startRunning()!
        session.commitConfiguration()

        if !session.isRunning {
            session.startRunning()
        }
        isRunning = session.isRunning

        delegate?.captureManager(
            self,
            didUpdateStatus: device.localizedName,
            fourCC: activeFourCCStr,
            resolution: activeResStr,
            warning: warningMsg
        )
    }

    private func attachMatchingAudioLocked(videoDeviceName: String) {
        let audioDiscovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.microphone, .external],
            mediaType: .audio,
            position: .unspecified
        )
        let keywords = ["cm716", "ugreen", "ms2130", "usb", "hdmi", "capture", "digital audio"]
        let lowerVideo = videoDeviceName.lowercased()

        let matchedAudio = audioDiscovery.devices.first(where: { dev in
            let name = dev.localizedName.lowercased()
            if !lowerVideo.isEmpty && (name.contains(lowerVideo) || lowerVideo.contains(name)) {
                return true
            }
            return keywords.contains(where: { name.contains($0) })
        })

        guard let audioDev = matchedAudio,
              let newAudioInput = try? AVCaptureDeviceInput(device: audioDev),
              session.canAddInput(newAudioInput) else {
            return
        }

        let previewOut = AVCaptureAudioPreviewOutput()
        previewOut.volume = isAudioMuted ? 0.0 : audioVolume
        if session.canAddOutput(previewOut) {
            session.addInput(newAudioInput)
            session.addOutput(previewOut)
            audioInput = newAudioInput
            audioPreviewOutput = previewOut
        }
    }

    private struct FormatCandidate {
        let format: AVCaptureDevice.Format
        let fps: Int
        let minFrameDuration: CMTime
        let maxFrameDuration: CMTime
        let score: Int
    }

    /// Scores every format advertised by CM716 (MS2130) to pick the cleanest uncompressed low-latency path.
    private func negotiateOptimalFormat(for device: AVCaptureDevice, profile: CaptureProfile) -> FormatCandidate? {
        let targetFPS = (profile == .hz30YUY2Lossless) ? 30.0 : 60.0
        var best: FormatCandidate?

        for fmt in device.formats {
            let dims = CMVideoFormatDescriptionGetDimensions(fmt.formatDescription)
            let subType = CMFormatDescriptionGetMediaSubType(fmt.formatDescription)
            let is1080p = (dims.width == 1920 && dims.height == 1080)
            let is720p = (dims.width == 1280 && dims.height == 720)
            let is1440p = (dims.width == 2560 && dims.height == 1440)

            // Pick the exact AVFrameRateRange closest to targetFPS (preserving MS2130's exact CMTime fractions!)
            var chosenRange: AVFrameRateRange?
            var chosenFPSDouble: Double = 0
            var supportsTarget = false

            for range in fmt.videoSupportedFrameRateRanges {
                if targetFPS >= range.minFrameRate - 0.5 && targetFPS <= range.maxFrameRate + 0.5 {
                    chosenRange = range
                    chosenFPSDouble = min(max(targetFPS, range.minFrameRate), range.maxFrameRate)
                    supportsTarget = true
                    break
                }
                if range.maxFrameRate > chosenFPSDouble && range.maxFrameRate <= 61.0 {
                    chosenRange = range
                    chosenFPSDouble = range.maxFrameRate
                }
            }
            guard let range = chosenRange else { continue }
            let chosenFPS = Int(round(chosenFPSDouble))
            if chosenFPS < 24 { continue }

            // Compute exact valid CMTime inside `range`
            let minDur: CMTime
            let maxDur: CMTime
            if abs(range.maxFrameRate - range.minFrameRate) < 0.5 {
                // Discrete hardware UVC interval (like UGREEN 25854's 1000000/60000240) — MUST pass verbatim!
                minDur = range.minFrameDuration
                maxDur = range.maxFrameDuration
            } else {
                // Continuous range (like Continuity Camera 1.0..60.0) — clamp requested duration strictly inside range
                let desired = CMTime(value: 1_000_000, timescale: CMTimeScale(round(chosenFPSDouble * 1_000_000)))
                let clamped = CMTimeClampToRange(
                    desired,
                    range: CMTimeRange(start: range.minFrameDuration, end: range.maxFrameDuration)
                )
                minDur = clamped
                maxDur = clamped
            }

            var score = 0
            // Resolution preference: 1080p is CM716 native maximum capture output
            if is1080p { score += 500 }
            else if is1440p { score += 450 }
            else if is720p { score += 200 }
            else { score += min(Int(dims.width) / 10, 150) }

            // Frame rate match
            if supportsTarget { score += 300 }
            else if chosenFPS >= 59 { score += 220 }
            else if chosenFPS >= 29 { score += 160 }

            // Pixel format scoring (heavily penalize MJPEG to avoid USB 2.0 / compression artifacts!)
            let isNV12 = (subType == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange || subType == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)
            let isYUY2 = (subType == kCVPixelFormatType_422YpCbCr8 || subType == kCVPixelFormatType_422YpCbCr8_yuvs)
            let isBGRA = (subType == kCVPixelFormatType_32BGRA)

            switch profile {
            case .hz60NV12Dedup:
                if isNV12 { score += 280 }
                else if isYUY2 { score += 260 }
                else if isBGRA { score += 220 }
                else { score -= 350 } // MJPEG penalty
            case .hz30YUY2Lossless:
                if isYUY2 { score += 320 }
                else if isNV12 { score += 270 }
                else if isBGRA { score += 220 }
                else { score -= 350 } // MJPEG penalty
            }

            if best == nil || score > best!.score {
                best = FormatCandidate(
                    format: fmt,
                    fps: chosenFPS,
                    minFrameDuration: minDur,
                    maxFrameDuration: maxDur,
                    score: score
                )
            }
        }
        return best
    }

    public func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        let fourCC = CVPixelBufferGetPixelFormatType(pixelBuffer)
        delegate?.captureManager(self, didOutput: pixelBuffer, presentationTime: pts, fourCC: fourCC)
    }

    public func fourCCString(_ code: OSType) -> String {
        let bytes: [UInt8] = [
            UInt8((code >> 24) & 0xFF),
            UInt8((code >> 16) & 0xFF),
            UInt8((code >> 8) & 0xFF),
            UInt8(code & 0xFF)
        ]
        return String(bytes: bytes, encoding: .ascii) ?? String(format: "0x%08X", code)
    }
}
