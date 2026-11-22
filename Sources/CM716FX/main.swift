import AppKit
import Foundation

// Entrypoint supporting both `--self-test` headless GPU verification and normal GUI app launch.
let args = CommandLine.arguments

if args.contains("--self-test") {
    do {
        var shaderPath: String? = nil
        if let idx = args.firstIndex(of: "--shader"), idx + 1 < args.count {
            shaderPath = args[idx + 1]
        }
        let engine = try MetalPipelineEngine(shaderSourcePath: shaderPath)
        let passed = engine.runHeadlessSelfTest(frameCount: 60)
        exit(passed ? 0 : 1)
    } catch {
        fputs("[ERROR] CM716FX self-test failed: \(error.localizedDescription)\n", stderr)
        exit(2)
    }
}

let app = NSApplication.shared
app.setActivationPolicy(.regular)
let delegate = AppDelegate()
app.delegate = delegate
app.run()
