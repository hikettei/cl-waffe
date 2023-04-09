// swift-tools-version:5.2

import PackageDescription

let package = Package(
    name: "MPSBridge",
    products: [
        .library(
            name: "MPSBridge",
            type: .dynamic,
            targets: ["MPSBridge"]),
    ],
    dependencies: [
        // Dependencies declare other packages that this package depends on.
        // .package(url: /* package url */, from: "1.0.0"),
    ],
    targets: [
        .target(
            name: "MPSBridge",
            dependencies: [])
    ]
)
