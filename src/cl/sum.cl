#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

#define WORK_GROUP_SIZE 256
__kernel void sum(__global const unsigned int *a, __global unsigned int *res, unsigned int n) {
    unsigned int globalId = get_global_id(0);
    unsigned int localId = get_local_id(0);
    __local unsigned int local_xs[WORK_GROUP_SIZE];
    local_xs[localId] = (globalId < n ? a[globalId] : 0);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (unsigned int block = WORK_GROUP_SIZE; block > 1; block /= 2) {
        if (2 * localId < block) {
            int x = local_xs[localId];
            int y = local_xs[localId + block / 2];
            local_xs[localId] = x + y;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (localId == 0) {
        res[globalId / WORK_GROUP_SIZE] = local_xs[0];
    }
}