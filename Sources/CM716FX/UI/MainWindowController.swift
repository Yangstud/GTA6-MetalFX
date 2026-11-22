import AppKit
import CoreVideo
import Metal
import QuartzCore

/// Dedicated low-latency Metal view backed directly by `CAMetalLayer` (double-buffered, zero AppKit redraw bloat).
public final class MetalDisplayView: NSView {
    public let metalLayer = CAMetalLayer()
    private let engine: MetalPipelineEngine
    private var displayLink: CVDisplayLink?
    private let renderQueue = DispatchQueue(label: "com.cm716fx.display.render", qos: .userInteractive)
    private var isRendering: Bool = false

    public init(frame frameRect: NSRect, engine: MetalPipelineEngine) {
        self.engine = engine
        super.init(frame: frameRect)
        wantsLayer = true
        layer = metalLayer

        metalLayer.device = engine.device
        metalLayer.pixelFormat = .bgra8Unorm
        metalLayer.framebufferOnly = false
        metalLayer.maximumDrawableCount = 2 // Double-buffering saves 1 full VSync frame (16.67ms) vs triple!
        metalLayer.presentsWithTransaction = false
        metalLayer.displaySyncEnabled = engine.config.vsyncEnabled
        metalLayer.contentsScale = NSScreen.main?.backingScaleFactor ?? 2.0
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    public override func layout() {
        super.layout()
        updateDrawableSize()
    }

    public override func viewDidChangeBackingProperties() {
        super.viewDidChangeBackingProperties()
        updateDrawableSize()
    }

    public func updateDrawableSize() {
        let scale = window?.backingScaleFactor ?? NSScreen.main?.backingScaleFactor ?? 2.0
        metalLayer.contentsScale = scale
        let w = max(1, Int(round(bounds.width * scale)))
        let h = max(1, Int(round(bounds.height * scale)))
        if Int(metalLayer.drawableSize.width) != w || Int(metalLayer.drawableSize.height) != h {
            metalLayer.drawableSize = CGSize(width: w, height: h)
        }
    }

    public func setVSyncEnabled(_ enabled: Bool) {
        metalLayer.displaySyncEnabled = enabled
    }

    public func startDisplayLink() {
        stopDisplayLink()
        var link: CVDisplayLink?
        CVDisplayLinkCreateWithActiveCGDisplays(&link)
        guard let dl = link else { return }

        let callback: CVDisplayLinkOutputCallback = { _, _, _, _, _, userInfo in
            guard let userInfo else { return kCVReturnSuccess }
            let view = Unmanaged<MetalDisplayView>.fromOpaque(userInfo).takeUnretainedValue()
            view.onVSyncTick()
            return kCVReturnSuccess
        }
        let selfPtr = Unmanaged.passUnretained(self).toOpaque()
        CVDisplayLinkSetOutputCallback(dl, callback, selfPtr)
        CVDisplayLinkStart(dl)
        displayLink = dl
    }

    public func stopDisplayLink() {
        if let dl = displayLink {
            CVDisplayLinkStop(dl)
            displayLink = nil
        }
    }

    private func onVSyncTick() {
        renderQueue.async { [weak self] in
            guard let self else { return }
            if self.isRendering { return }
            self.isRendering = true
            defer { self.isRendering = false }

            autoreleasepool {
                guard let drawable = self.metalLayer.nextDrawable() else { return }
                let vp = self.metalLayer.drawableSize
                self.engine.renderNextDisplayFrame(into: drawable, viewportSize: vp)
            }
        }
    }

    deinit {
        stopDisplayLink()
    }
}

/// Main window with borderless fullscreen toggle, keyboard shortcuts, and frosted telemetry HUD.
public final class MainWindowController: NSWindowController, NSWindowDelegate {
    private let engine: MetalPipelineEngine
    private let capture: CM716CaptureManager
    private let metalView: MetalDisplayView

    private let hudContainer = NSVisualEffectView()
    private let hudTitleLabel = NSTextField(labelWithString: "")
    private let hudBodyLabel = NSTextField(labelWithString: "")
    private let hudWarningLabel = NSTextField(labelWithString: "")
    private let hudHotkeyLabel = NSTextField(labelWithString: "")
    private var hudTimer: Timer?

    private var savedStyleMask: NSWindow.StyleMask = [.titled, .closable, .miniaturizable, .resizable]
    private var savedFrame: NSRect = NSRect(x: 120, y: 120, width: 1280, height: 720)
    private var isBorderlessFullscreen: Bool = false

    public var onConfigChanged: ((PipelineConfig) -> Void)?

    public init(engine: MetalPipelineEngine, capture: CM716CaptureManager) {
        self.engine = engine
        self.capture = capture

        let initialRect = NSRect(x: 100, y: 100, width: 1280, height: 720)
        let win = KeyHandlingWindow(
            contentRect: initialRect,
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        win.title = "CM716FX — MetalFX 空间超分 + Apple AI 帧生成引擎"
        win.backgroundColor = .black
        win.contentAspectRatio = NSSize(width: 16, height: 9)
        win.minSize = NSSize(width: 640, height: 360)
        win.isMovableByWindowBackground = true

        self.metalView = MetalDisplayView(frame: NSRect(origin: .zero, size: initialRect.size), engine: engine)
        self.metalView.autoresizingMask = [.width, .height]

        super.init(window: win)
        win.delegate = self
        win.onKeyDown = { [weak self] event in
            return self?.handleKeyEvent(event) ?? false
        }

        let content = NSView(frame: NSRect(origin: .zero, size: initialRect.size))
        content.wantsLayer = true
        content.layer?.backgroundColor = NSColor.black.cgColor
        content.addSubview(metalView)
        win.contentView = content

        setupHUD(in: content)
        metalView.startDisplayLink()
        startHUDTimer()
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    private func setupHUD(in container: NSView) {
        hudContainer.translatesAutoresizingMaskIntoConstraints = false
        hudContainer.material = .hudWindow
        hudContainer.blendingMode = .withinWindow
        hudContainer.state = .active
        hudContainer.wantsLayer = true
        hudContainer.layer?.cornerRadius = 12
        hudContainer.layer?.borderWidth = 1
        hudContainer.layer?.borderColor = NSColor.white.withAlphaComponent(0.16).cgColor
        container.addSubview(hudContainer)

        hudTitleLabel.font = .monospacedSystemFont(ofSize: 13, weight: .bold)
        hudTitleLabel.textColor = NSColor(calibratedRed: 0.40, green: 0.92, blue: 1.0, alpha: 1.0)
        hudTitleLabel.translatesAutoresizingMaskIntoConstraints = false

        hudBodyLabel.font = .monospacedSystemFont(ofSize: 11.5, weight: .medium)
        hudBodyLabel.textColor = .white
        hudBodyLabel.maximumNumberOfLines = 12
        hudBodyLabel.translatesAutoresizingMaskIntoConstraints = false

        hudWarningLabel.font = .monospacedSystemFont(ofSize: 11.5, weight: .bold)
        hudWarningLabel.textColor = NSColor(calibratedRed: 1.0, green: 0.42, blue: 0.36, alpha: 1.0)
        hudWarningLabel.maximumNumberOfLines = 3
        hudWarningLabel.translatesAutoresizingMaskIntoConstraints = false

        hudHotkeyLabel.font = .monospacedSystemFont(ofSize: 10.5, weight: .regular)
        hudHotkeyLabel.textColor = NSColor.white.withAlphaComponent(0.72)
        hudHotkeyLabel.maximumNumberOfLines = 4
        hudHotkeyLabel.stringValue = "[1-5]预设  [G]切插帧  [U]切超分  [D]去重帧  [C]60Hz/30Hz采  [S]CAS锐化\n[M]Adrenaline洁净喂帧  [R]色域  [A]静音  [T]内置30fps测试源  [F]全屏  [H]藏HUD"
        hudHotkeyLabel.translatesAutoresizingMaskIntoConstraints = false

        let stack = NSStackView(views: [hudTitleLabel, hudBodyLabel, hudWarningLabel, hudHotkeyLabel])
        stack.orientation = .vertical
        stack.alignment = .leading
        stack.spacing = 6
        stack.translatesAutoresizingMaskIntoConstraints = false
        hudContainer.addSubview(stack)

        NSLayoutConstraint.activate([
            hudContainer.leadingAnchor.constraint(equalTo: container.leadingAnchor, constant: 16),
            hudContainer.topAnchor.constraint(equalTo: container.topAnchor, constant: 16),
            hudContainer.widthAnchor.constraint(lessThanOrEqualToConstant: 580),

            stack.leadingAnchor.constraint(equalTo: hudContainer.leadingAnchor, constant: 14),
            stack.trailingAnchor.constraint(equalTo: hudContainer.trailingAnchor, constant: -14),
            stack.topAnchor.constraint(equalTo: hudContainer.topAnchor, constant: 12),
            stack.bottomAnchor.constraint(equalTo: hudContainer.bottomAnchor, constant: -12)
        ])
    }

    private func startHUDTimer() {
        hudTimer = Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) { [weak self] _ in
            self?.refreshHUD()
        }
    }

    public func refreshHUD() {
        let cfg = engine.config
        hudContainer.isHidden = !cfg.showHUD || cfg.adrenalineCompanionMode
        if hudContainer.isHidden { return }

        let snap = engine.telemetrySnapshot()
        hudTitleLabel.stringValue = "CM716FX  |  \(snap.hardwareCapabilitiesSummary)"

        let dedupTag = cfg.enableDedup
            ? String(format: "ON (剔除 %.0f%% | MAD %.4f / Peak %.4f)", snap.droppedDuplicateRatio * 100, snap.lastGlobalMAD, snap.lastPeakBlockMAD)
            : "OFF"

        let lines = [
            "采集设备 : \(snap.deviceName)",
            "采集信号 : \(snap.inputResolution) [\(snap.pixelFormatFourCC)]  →  输出 \(snap.outputResolution) | \(cfg.colorRange == .limited ? "Limited" : "Full")",
            String(format: "帧率链路 : 采集 %.1f fps → 去重唯一 %.1f fps → 最终显示 %.1f fps", snap.captureFPS, snap.uniqueFPS, snap.outputFPS),
            "重复帧过滤: \(dedupTag) | 硬切保护: \(snap.sceneCutCount) 次",
            "帧生成引擎: \(snap.activeFrameGenLabel)",
            "超分与锐化: \(snap.activeUpscalerLabel)",
            String(format: "GPU 耗时 : 转换去重 %.2fms + 插帧 %.2fms + 超分呈现 %.2fms = 总计 %.2fms",
                   snap.gpuConvertDedupMs, snap.gpuFrameGenMs, snap.gpuUpscalePresentMs, snap.totalPipelineMs)
        ]
        hudBodyLabel.stringValue = lines.joined(separator: "\n")

        if let warn = snap.usbBandwidthWarning {
            hudWarningLabel.isHidden = false
            hudWarningLabel.stringValue = warn
        } else {
            hudWarningLabel.isHidden = true
            hudWarningLabel.stringValue = ""
        }
    }

    public func applyConfig(_ newConfig: PipelineConfig) {
        let old = engine.config
        engine.config = newConfig
        metalView.setVSyncEnabled(newConfig.vsyncEnabled)

        if old.captureProfile != newConfig.captureProfile {
            capture.setProfile(newConfig.captureProfile)
        }
        if old.audioMuted != newConfig.audioMuted || old.audioVolume != newConfig.audioVolume {
            capture.setAudio(muted: newConfig.audioMuted, volume: newConfig.audioVolume)
        }
        if old.syntheticTestMode != newConfig.syntheticTestMode {
            engine.setSyntheticTestMode(enabled: newConfig.syntheticTestMode)
            if !newConfig.syntheticTestMode {
                capture.start(deviceID: nil, profile: newConfig.captureProfile)
            }
        }

        // In Adrenaline Clean-Feed Mode, make the window borderless clean 16:9 with hidden HUD
        if old.adrenalineCompanionMode != newConfig.adrenalineCompanionMode {
            updateCompanionWindowChrome(active: newConfig.adrenalineCompanionMode)
        }

        refreshHUD()
        onConfigChanged?(newConfig)
    }

    private func updateCompanionWindowChrome(active: Bool) {
        guard let win = window, !isBorderlessFullscreen else { return }
        if active {
            win.titlebarAppearsTransparent = true
            win.titleVisibility = .hidden
            win.styleMask.insert(.fullSizeContentView)
        } else {
            win.titlebarAppearsTransparent = false
            win.titleVisibility = .visible
            win.styleMask.remove(.fullSizeContentView)
        }
        metalView.updateDrawableSize()
    }

    public func toggleBorderlessFullscreen() {
        guard let win = window, let screen = win.screen ?? NSScreen.main else { return }
        if !isBorderlessFullscreen {
            savedStyleMask = win.styleMask
            savedFrame = win.frame
            win.styleMask = [.borderless]
            win.level = .mainMenu + 1
            win.setFrame(screen.frame, display: true)
            isBorderlessFullscreen = true
        } else {
            win.level = .normal
            win.styleMask = savedStyleMask
            win.setFrame(savedFrame, display: true)
            isBorderlessFullscreen = false
        }
        metalView.updateDrawableSize()
    }

    private func handleKeyEvent(_ event: NSEvent) -> Bool {
        guard let chars = event.charactersIgnoringModifiers?.lowercased(), !chars.isEmpty else { return false }
        var cfg = engine.config

        switch chars {
        case "1":
            cfg.apply(preset: .m2Balanced60); applyConfig(cfg); return true
        case "2":
            cfg.apply(preset: .m2Combo4K60); applyConfig(cfg); return true
        case "3":
            cfg.apply(preset: .m2ProMotion120); applyConfig(cfg); return true
        case "4":
            cfg.apply(preset: .adrenalineCleanFeed); applyConfig(cfg); return true
        case "5":
            cfg.apply(preset: .rawPassthrough); applyConfig(cfg); return true
        case "g", " ":
            let all = FrameGenBackend.allCases
            if let idx = all.firstIndex(of: cfg.frameGen) {
                cfg.frameGen = all[(idx + 1) % all.count]
                cfg.adrenalineCompanionMode = false
                applyConfig(cfg)
            }
            return true
        case "u":
            let all = UpscalerBackend.allCases
            if let idx = all.firstIndex(of: cfg.upscaler) {
                cfg.upscaler = all[(idx + 1) % all.count]
                cfg.adrenalineCompanionMode = false
                applyConfig(cfg)
            }
            return true
        case "d":
            cfg.enableDedup.toggle(); applyConfig(cfg); return true
        case "c":
            cfg.captureProfile = (cfg.captureProfile == .hz60NV12Dedup) ? .hz30YUY2Lossless : .hz60NV12Dedup
            applyConfig(cfg); return true
        case "r":
            cfg.colorRange = (cfg.colorRange == .limited) ? .full : .limited
            applyConfig(cfg); return true
        case "s":
            let steps: [Float] = [0.0, 0.25, 0.50, 0.75]
            let next = steps.first(where: { $0 > cfg.casSharpness + 0.05 }) ?? 0.0
            cfg.casSharpness = next
            applyConfig(cfg); return true
        case "m":
            if cfg.adrenalineCompanionMode {
                cfg.apply(preset: .m2Balanced60)
            } else {
                cfg.apply(preset: .adrenalineCleanFeed)
            }
            applyConfig(cfg); return true
        case "a":
            cfg.audioMuted.toggle(); applyConfig(cfg); return true
        case "t":
            cfg.syntheticTestMode.toggle(); applyConfig(cfg); return true
        case "h", "\t":
            cfg.showHUD.toggle(); applyConfig(cfg); return true
        case "f":
            toggleBorderlessFullscreen(); return true
        case "\u{1b}": // Escape exits borderless fullscreen
            if isBorderlessFullscreen { toggleBorderlessFullscreen(); return true }
            return false
        default:
            return false
        }
    }

    public func windowWillClose(_ notification: Notification) {
        hudTimer?.invalidate()
        metalView.stopDisplayLink()
        capture.stop()
        engine.setSyntheticTestMode(enabled: false)
    }
}

/// Custom NSWindow subclass that routes single-key hotkeys cleanly without system beep.
private final class KeyHandlingWindow: NSWindow {
    var onKeyDown: ((NSEvent) -> Bool)?

    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { true }

    override func keyDown(with event: NSEvent) {
        if let handler = onKeyDown, handler(event) {
            return
        }
        super.keyDown(with: event)
    }
}
