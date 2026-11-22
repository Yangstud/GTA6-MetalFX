import AppKit
import AVFoundation
import CoreMedia
import CoreVideo

public final class AppDelegate: NSObject, NSApplicationDelegate, CM716CaptureDelegate {
    private var engine: MetalPipelineEngine!
    private var capture: CM716CaptureManager!
    private var windowController: MainWindowController!
    private var devicesMenu: NSMenu?

    public func applicationDidFinishLaunching(_ notification: Notification) {
        do {
            engine = try MetalPipelineEngine()
        } catch {
            let alert = NSAlert()
            alert.messageText = "CM716FX 初始化失败"
            alert.informativeText = error.localizedDescription
            alert.runModal()
            NSApp.terminate(nil)
            return
        }

        capture = CM716CaptureManager()
        capture.delegate = self

        windowController = MainWindowController(engine: engine, capture: capture)
        windowController.showWindow(nil)
        windowController.window?.makeKeyAndOrderFront(nil)

        buildMainMenu()
        NSApp.activate(ignoringOtherApps: true)

        // Request macOS camera permission & auto-connect CM716; if no capture card attached, start synthetic test!
        let status = AVCaptureDevice.authorizationStatus(for: .video)
        switch status {
        case .authorized:
            startCaptureOrFallbackDemo()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                DispatchQueue.main.async {
                    if granted {
                        self?.startCaptureOrFallbackDemo()
                    } else {
                        self?.enableSyntheticFallback(reason: "未授摄像头权限，已自动切至内置 30fps 测试画面（可去系统设置授权后点菜单重连）")
                    }
                }
            }
        default:
            enableSyntheticFallback(reason: "系统未授权访问采集卡（隐私→摄像头），已切至内置 30fps 测试画面")
        }
    }

    private func startCaptureOrFallbackDemo() {
        let devices = capture.listVideoDevices()
        refreshDevicesMenu(devices: devices)
        if let cm716 = devices.first(where: { $0.isLikelyCM716 }) {
            capture.start(deviceID: cm716.uniqueID, profile: engine.config.captureProfile)
        } else {
            enableSyntheticFallback(reason: nil)
        }
    }

    private func enableSyntheticFallback(reason: String?) {
        var cfg = engine.config
        cfg.syntheticTestMode = true
        windowController.applyConfig(cfg)
        if let reason {
            engine.updateCaptureStatus(deviceName: reason, fourCC: "SYNTH", resolution: "1920x1080@60(30u)", warning: nil)
        }
    }

    public func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }

    // MARK: - CM716CaptureDelegate

    public func captureManager(
        _ manager: CM716CaptureManager,
        didOutput pixelBuffer: CVPixelBuffer,
        presentationTime: CMTime,
        fourCC: OSType
    ) {
        engine.ingestCaptureBuffer(pixelBuffer, pts: presentationTime, fourCC: fourCC)
    }

    public func captureManager(
        _ manager: CM716CaptureManager,
        didUpdateStatus deviceName: String,
        fourCC: String,
        resolution: String,
        warning: String?
    ) {
        engine.updateCaptureStatus(deviceName: deviceName, fourCC: fourCC, resolution: resolution, warning: warning)
    }

    // MARK: - macOS Menu Bar

    private func buildMainMenu() {
        let mainMenu = NSMenu()

        // App Menu
        let appItem = NSMenuItem()
        let appMenu = NSMenu()
        appMenu.addItem(withTitle: "退出 CM716FX", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        appItem.submenu = appMenu
        mainMenu.addItem(appItem)

        // Capture Device Menu
        let devItem = NSMenuItem()
        let devMenu = NSMenu(title: "采集设备")
        devicesMenu = devMenu
        devItem.submenu = devMenu
        mainMenu.addItem(devItem)
        refreshDevicesMenu(devices: capture.listVideoDevices())

        // Presets Menu
        let presetItem = NSMenuItem()
        let presetMenu = NSMenu(title: "场景预设")
        for (idx, preset) in AppPreset.allCases.enumerated() {
            let item = NSMenuItem(title: preset.rawValue, action: #selector(selectPreset(_:)), keyEquivalent: "\(idx + 1)")
            item.target = self
            item.tag = idx
            presetMenu.addItem(item)
        }
        presetMenu.addItem(.separator())
        let synthItem = NSMenuItem(title: "切换内置 1080p 30fps 测试画面 (T)", action: #selector(toggleSynth), keyEquivalent: "t")
        synthItem.target = self
        presetMenu.addItem(synthItem)
        let fsItem = NSMenuItem(title: "无边框全屏 (F)", action: #selector(toggleFullscreen), keyEquivalent: "f")
        fsItem.target = self
        presetMenu.addItem(fsItem)
        presetItem.submenu = presetMenu
        mainMenu.addItem(presetItem)

        // Engine Menu
        let engItem = NSMenuItem()
        let engMenu = NSMenu(title: "插帧与超分")
        for (idx, fg) in FrameGenBackend.allCases.enumerated() {
            let item = NSMenuItem(title: "插帧: \(fg.rawValue)", action: #selector(selectFG(_:)), keyEquivalent: "")
            item.target = self
            item.tag = idx
            engMenu.addItem(item)
        }
        engMenu.addItem(.separator())
        for (idx, sr) in UpscalerBackend.allCases.enumerated() {
            let item = NSMenuItem(title: "超分: \(sr.rawValue)", action: #selector(selectSR(_:)), keyEquivalent: "")
            item.target = self
            item.tag = idx
            engMenu.addItem(item)
        }
        engItem.submenu = engMenu
        mainMenu.addItem(engItem)

        NSApp.mainMenu = mainMenu
    }

    private func refreshDevicesMenu(devices: [CaptureDeviceInfo]) {
        guard let devMenu = devicesMenu else { return }
        devMenu.removeAllItems()

        let refreshItem = NSMenuItem(title: "重新扫描 CM716 / 采集卡", action: #selector(rescanDevices), keyEquivalent: "r")
        refreshItem.target = self
        devMenu.addItem(refreshItem)
        devMenu.addItem(.separator())

        if devices.isEmpty {
            let empty = NSMenuItem(title: "（未检测到任何视频设备）", action: nil, keyEquivalent: "")
            empty.isEnabled = false
            devMenu.addItem(empty)
        } else {
            for dev in devices {
                let tag = dev.isLikelyCM716 ? "★ [采集卡] " : ""
                let item = NSMenuItem(title: "\(tag)\(dev.localizedName)", action: #selector(selectDevice(_:)), keyEquivalent: "")
                item.target = self
                item.representedObject = dev.uniqueID
                devMenu.addItem(item)
            }
        }
    }

    @objc private func rescanDevices() {
        let devices = capture.listVideoDevices()
        refreshDevicesMenu(devices: devices)
        if let cm716 = devices.first(where: { $0.isLikelyCM716 }) {
            var cfg = engine.config
            cfg.syntheticTestMode = false
            windowController.applyConfig(cfg)
            capture.start(deviceID: cm716.uniqueID, profile: cfg.captureProfile)
        }
    }

    @objc private func selectDevice(_ sender: NSMenuItem) {
        guard let uid = sender.representedObject as? String else { return }
        var cfg = engine.config
        cfg.syntheticTestMode = false
        windowController.applyConfig(cfg)
        capture.start(deviceID: uid, profile: cfg.captureProfile)
    }

    @objc private func selectPreset(_ sender: NSMenuItem) {
        let all = AppPreset.allCases
        guard sender.tag >= 0 && sender.tag < all.count else { return }
        var cfg = engine.config
        cfg.apply(preset: all[sender.tag])
        windowController.applyConfig(cfg)
    }

    @objc private func selectFG(_ sender: NSMenuItem) {
        let all = FrameGenBackend.allCases
        guard sender.tag >= 0 && sender.tag < all.count else { return }
        var cfg = engine.config
        cfg.frameGen = all[sender.tag]
        cfg.adrenalineCompanionMode = false
        windowController.applyConfig(cfg)
    }

    @objc private func selectSR(_ sender: NSMenuItem) {
        let all = UpscalerBackend.allCases
        guard sender.tag >= 0 && sender.tag < all.count else { return }
        var cfg = engine.config
        cfg.upscaler = all[sender.tag]
        cfg.adrenalineCompanionMode = false
        windowController.applyConfig(cfg)
    }

    @objc private func toggleSynth() {
        var cfg = engine.config
        cfg.syntheticTestMode.toggle()
        windowController.applyConfig(cfg)
    }

    @objc private func toggleFullscreen() {
        windowController.toggleBorderlessFullscreen()
    }
}
