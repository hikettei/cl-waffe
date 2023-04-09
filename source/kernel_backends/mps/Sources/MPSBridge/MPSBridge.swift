
import Metal
import MetalPerformanceShaders
import Accelerate
import Foundation

let metallib =  "\(#file.replacingOccurrences(of: "/MPSBridge.swift", with: ""))/Shaders.metallib"

// Initialized Device.
// device=MLTDevice
@available(macOS 10.13, *)
let device = MTLCreateSystemDefaultDevice()!,
    commandQueue = device.makeCommandQueue()!,
    defaultLibrary = try! device.makeLibrary(filepath: metallib)
 

@available(macOS 10.13, *)
// mps_2dfgemm(alpha, a, b, beta, c, m, n, k, transpose_a, transpose_b)
// [m, k] [k, n] -> [m, n]
@_cdecl("mps_2dfgemm")
public func mps_2dfgemm(alpha: Double,
                        a: UnsafePointer<Float>,
                        b: UnsafePointer<Float>,
                        beta: Double, 
                        c: UnsafeMutablePointer<Float>,
                        m: Int,
                        n: Int,
                        k: Int,
                        Transpose_A: Bool,
                        Transpose_B: Bool) -> Int{
    let matrixMultiplication = MPSMatrixMultiplication(device: device,
                                                       transposeLeft: Transpose_A,
                                                       transposeRight: Transpose_B,
                                                       resultRows: m,
                                                       resultColumns: n,
                                                       interiorColumns: k,
                                                       alpha: alpha,
                                                       beta: beta)

    let size_a = m * k
    let size_b = k * n
    let size_c = m * n

    let commandBuffer = commandQueue.makeCommandBuffer()!

    let bufferA = device.makeBuffer(bytes: a, length: size_a * MemoryLayout<Float>.stride, options: [])!
    let bufferB = device.makeBuffer(bytes: b, length: size_b * MemoryLayout<Float>.stride, options: [])!
    let bufferC = device.makeBuffer(bytes: c, length: size_c * MemoryLayout<Float>.stride, options: [])!

    let descA = MPSMatrixDescriptor(rows: m, columns: k, rowBytes: k * MemoryLayout<Float>.stride, dataType: .float32)
    let descB = MPSMatrixDescriptor(rows: k, columns: n, rowBytes: k * MemoryLayout<Float>.stride, dataType: .float32)
    let descC = MPSMatrixDescriptor(rows: m, columns: n, rowBytes: m * MemoryLayout<Float>.stride, dataType: .float32)

    let matrixA = MPSMatrix(buffer: bufferA, descriptor: descA)
    let matrixB = MPSMatrix(buffer: bufferB, descriptor: descB)
    let matrixC = MPSMatrix(buffer: bufferC, descriptor: descC)

    matrixMultiplication.encode(commandBuffer: commandBuffer, leftMatrix: matrixA, rightMatrix: matrixB, resultMatrix: matrixC)
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    c.initialize(from: bufferC.contents().assumingMemoryBound(to: Float.self), count: m * n)
    return 1
}
