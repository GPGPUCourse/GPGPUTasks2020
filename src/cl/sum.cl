#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 8

__kernel void sum(__global const uint *as, __global uint *result) {
    const uint globalId = get_global_id(0);
    const uint localId = get_local_id(0);
    const uint localSize = 128; // get_local_size(0)
    __local uint localAs[localSize];
    localAs[localId] = as[globalId];
    for (uint offset = localSize / 2; offset != 0; offset /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localId < offset) {
            localAs[localId] += localAs[localId + offset];
        }
    }
    if (localId == 0) {
        atomic_add(result, localAs[0]);
    }
}