
xcrun -sdk macosx metal -c ./source/kernel_backends/mps/Sources/MPSBridge/kernel.metal -o ./source/kernel_backends/mps/Sources/MPSBridge/kernel.air
xcrun -sdk macosx metallib ./source/kernel_backends/mps/Sources/MPSBridge/kernel.air -o ./source/kernel_backends/mps/Sources/MPSBridge/kernel.metallib
