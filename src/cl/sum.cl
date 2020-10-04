#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 256
#define VALUES_PER_WORK_ITEM 64
////      mixed   sum_gpu_4    and     sum_gpu_6
__kernel void sum(__global const unsigned int* xs,
                  __global unsigned int* out,
                  unsigned int n) {

    unsigned int localId = get_local_id(0);
    unsigned int groupId = get_group_id(0);
    __local unsigned int local_xs[WORK_GROUP_SIZE];

    local_xs[localId] = 0;
    // add to local memory
    for (size_t i = 0; i < VALUES_PER_WORK_ITEM; ++i) {
        unsigned int idx = groupId * WORK_GROUP_SIZE * VALUES_PER_WORK_ITEM + i * WORK_GROUP_SIZE + localId;
        local_xs[localId] += xs[idx] * (idx < n);
    }
    // tree sum
    barrier(CLK_LOCAL_MEM_FENCE);
    for (size_t nvalues = WORK_GROUP_SIZE; nvalues > 1; nvalues /= 2) {
        if (2 * localId < nvalues) {
            unsigned int a = local_xs[localId];
            unsigned int b = local_xs[localId + nvalues/2];
            local_xs[localId] = a + b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // atomic adding in 1st thread
    if (localId == 0) {
        atomic_add(out, local_xs[0]);
    }
}
