
#include <metal_stdlib>

using namespace metal;

kernel void sin(const device float *inVector [[ buffer(0) ]],
		device float *outVector [[ buffer(1) ]],
		uint id [[ thread_position_in_grid ]]) {
  outVector[id] = sin(inVector[id]);
}

kernel void cos(const device float *inVector [[ buffer(0) ]],
		device float *outVector [[ buffer(1) ]],
		uint id [[ thread_position_in_grid ]]) {
  outVector[id] = cos(inVector[id]);
}

kernel void tan(const device float *inVector [[ buffer(0) ]],
		device float *outVector [[ buffer(1) ]],
		uint id [[ thread_position_in_grid ]]) {
  outVector[id] = tan(inVector[id]);
}
