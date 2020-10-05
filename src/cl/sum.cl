#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 256

__kernel void sum_atomic(__global const unsigned int *as, __global unsigned int *sum, unsigned int n) {
    unsigned int g = get_global_id(0);
    unsigned int l = get_local_id(0);

    __local unsigned int loc[WORK_GROUP_SIZE];
    loc[l] = (g >= n) ? 0 : as[g];

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int part = WORK_GROUP_SIZE / 2; part >= 1; part /= 2) {
        if (l < part) {
            unsigned int a = loc[l];
            unsigned int b = loc[l + part];
            loc[l] = a + b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (l == 0) {
        atomic_add(sum, loc[0]);
    }
}

__kernel void sum_swap(__global const unsigned int *as, __global unsigned int *as_out, unsigned int n) {
    unsigned int g = get_global_id(0);
    unsigned int l = get_local_id(0);
    unsigned int j = get_group_id(0);

    __local unsigned int loc[WORK_GROUP_SIZE];
    loc[l] = (g >= n) ? 0 : as[g];

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int part = WORK_GROUP_SIZE / 2; part >= 1; part /= 2) {
        if (l < part) {
            unsigned int a = loc[l];
            unsigned int b = loc[l + part];
            loc[l] = a + b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (l == 0) {
        as_out[j] = loc[0];
    }
}