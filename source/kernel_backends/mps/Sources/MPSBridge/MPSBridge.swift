
import Metal
import MetalPerformanceShaders
import Accelerate
import Foundation

let metallib =  "\(#file.replacingOccurrences(of: "/MPSBridge.swift", with: ""))/Shaders.metallib"

 @available(macOS 10.13, *)
 let device = MTLCreateSystemDefaultDevice()!,
     commandQueue = device.makeCommandQueue()!,
     defaultLibrary = try! device.makeLibrary(filepath: metallib)
 
