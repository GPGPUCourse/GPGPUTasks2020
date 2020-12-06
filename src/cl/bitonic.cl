#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

#define WORK_GROUP_SIZE 256

__kernel void bitonic_local_up(__global float *as, unsigned int n) {
    int globalId = get_global_id(0);
    int localId = get_local_id(0);
    __local float buf[WORK_GROUP_SIZE];
    buf[localId] = (globalId < n ? as[globalId] : INFINITY);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (unsigned int b = 2; b <= WORK_GROUP_SIZE; b *= 2) {
        bool order = ((globalId & b) == 0);
        for (unsigned int sb = b; sb > 1; sb /= 2) {
            if ((localId & (sb - 1)) < sb / 2) {
                float x = buf[localId];
                float y = buf[localId + sb / 2];
                if ((x < y) != order) {
                    buf[localId] = y;
                    buf[localId + sb / 2] = x;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    if (globalId < n) {
        as[globalId] = buf[localId];
    }
}

__kernel void bitonic_local_down(__global float *as, unsigned int n, unsigned int block) {
    int globalId = get_global_id(0);
    int localId = get_local_id(0);
    __local float buf[WORK_GROUP_SIZE];
    buf[localId] = (globalId < n ? as[globalId] : INFINITY);
    barrier(CLK_LOCAL_MEM_FENCE);
    bool order = ((globalId & block) == 0);
    for (unsigned int sb = WORK_GROUP_SIZE; sb > 1; sb /= 2) {
        if ((localId & (sb - 1)) < sb / 2) {
            float x = buf[localId];
            float y = buf[localId + sb / 2];
            if ((x < y) != order) {
                buf[localId] = y;
                buf[localId + sb / 2] = x;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (globalId < n) {
        as[globalId] = buf[localId];
    }
}


__kernel void bitonic(__global float *as, unsigned int n, unsigned int block, unsigned int sb) {
    int globalId = get_global_id(0);
    bool order = ((globalId & block) == 0);
    if (globalId + sb / 2 < n && (globalId & (sb - 1)) < sb / 2) {
        float x = as[globalId];
        float y = as[globalId + sb / 2];
        if ((x < y) != order) {
            as[globalId] = y;
            as[globalId + sb / 2] = x;
        }
    }
}
