// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "ZImage",
    platforms: [.macOS(.v13)],
    products: [
        .executable(name: "ZImage", targets: ["ZImage"])
    ],
    targets: [
        .executableTarget(
            name: "ZImage",
            path: "Sources/ZImage",
            swiftSettings: [
                .unsafeFlags([
                    "-I", "/Users/htr/Documents/develeop/rust/burn/z-image-app",
                    "-L", "/Users/htr/Documents/develeop/rust/burn/z-image-app/target/release",
                    "-lz_image_ffi"
                ])
            ],
            linkerSettings: [
                .unsafeFlags([
                    "-L", "/Users/htr/Documents/develeop/rust/burn/z-image-app/target/release",
                    "-lz_image_ffi",
                    "-Xlinker", "-rpath", "-Xlinker", "@executable_path/../Frameworks"
                ])
            ]
        )
    ]
)
