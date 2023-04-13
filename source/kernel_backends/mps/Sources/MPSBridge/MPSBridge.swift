
import Metal
import MetalPerformanceShaders
import Accelerate
import Foundation

let metallib =  "\(#file.replacingOccurrences(of: "/MPSBridge.swift", with: ""))/kernel.metallib"

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
    return 0
}


@available(macOS 10.13, *)
@_cdecl("fsin_on_mps")
public func fsin_on_mps(input: UnsafePointer<Float>, output: UnsafeMutablePointer<Float>, count: Int) -> Int {
    return compute_1dfunc_kernel(functionName: "fsin", input: input, output: output, count: count)
}

@available(macOS 10.13, *)
func compute_1dfunc_kernel(functionName: String, input: UnsafePointer<Float>, output: UnsafeMutablePointer<Float>, count: Int) -> Int {
    do {
        let inputBuffer = UnsafeRawPointer(input)
        let commandBuffer = commandQueue.makeCommandBuffer()!
        let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!

        let Function = defaultLibrary.makeFunction(name: functionName)!
        let computePipelineState = try device.makeComputePipelineState(function: Function)
        computeCommandEncoder.setComputePipelineState(computePipelineState)

        let inputByteLength = count*MemoryLayout<Float>.size

        let inVectorBuffer = device.makeBuffer(bytes: inputBuffer, length: inputByteLength, options: [])

        computeCommandEncoder.setBuffer(inVectorBuffer, offset: 0, index: 0)

        let resultRef = UnsafeMutablePointer<Float>.allocate(capacity: count)
        let outVectorBuffer = device.makeBuffer(bytes: resultRef, length: inputByteLength, options: [])

        computeCommandEncoder.setBuffer(outVectorBuffer, offset: 0, index: 1)
        
        let maxTotalThreadsPerThreadgroup = computePipelineState.maxTotalThreadsPerThreadgroup
        let threadExecutionWidth = computePipelineState.threadExecutionWidth
        let width  = maxTotalThreadsPerThreadgroup / threadExecutionWidth * threadExecutionWidth
        let height = 1
        let depth  = 1
        
        // 1D
        let threadsPerGroup = MTLSize(width:width, height: height, depth: depth)
        let numThreadgroups = MTLSize(width: (count + width - 1) / width, height: 1, depth: 1)
        
        computeCommandEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
        computeCommandEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // unsafe bitcast and assigin result pointer to output
        output.initialize(from: outVectorBuffer!.contents().assumingMemoryBound(to: Float.self), count: count)
        
        free(resultRef)

        return 0
    } catch {
        
        return 1
    }
}
