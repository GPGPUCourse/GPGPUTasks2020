#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6
#define WORK_GROUP_SIZE 256

__kernel void sum(__global const unsigned int* from,
                  __global       unsigned int* to,
                  const unsigned int n)
{
    const unsigned int global_index = get_global_id(0);
    const unsigned int local_index = get_local_id(0);
    const unsigned int group_index = get_group_id(0);

    __local unsigned int local_from[WORK_GROUP_SIZE];
    if (global_index < n) {
        local_from[local_index] = from[global_index];
    } else {
        local_from[local_index] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_index == 0) {
        unsigned int sums = 0;
        for (int i = 0; i < WORK_GROUP_SIZE; ++i) {
            sums += local_from[i];
        }
        to[group_index] = sums;
    }
}

__kernel void sum_tree(__global const unsigned int* from,
                         __global       unsigned int* res,
                         const unsigned int n)
{
    const unsigned int global_index = get_global_id(0);
    const unsigned int local_index = get_local_id(0);

    __local unsigned int local_from[WORK_GROUP_SIZE];
    if (global_index < n) {
        local_from[local_index] = from[global_index];
    } else {
        local_from[local_index] = 0;
    }
    if (global_index == 0) {
        *res = 0;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    for (int step = WORK_GROUP_SIZE; step > 0; step = step / 2) {
        if (local_index < step / 2) {
            local_from[local_index] += local_from[local_index + step / 2];
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    if (local_index == 0) {
        atomic_add(res, local_from[0]);
    }
}

__kernel void sum_atomic(__global const unsigned int* from,
                  __global       unsigned int* res,
                  const unsigned int n)
{
    const unsigned int global_index = get_global_id(0);
    const unsigned int local_index = get_local_id(0);

    __local unsigned int local_from[WORK_GROUP_SIZE];
    if (global_index < n) {
        local_from[local_index] = from[global_index];
    } else {
        local_from[local_index] = 0;
    }
    if (global_index == 0) {
        *res = 0;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    if (local_index == 0) {
        unsigned int sums = 0;
        for (int i = 0; i < WORK_GROUP_SIZE; ++i) {
            sums += local_from[i];
        }
        atomic_add(res, sums);
    }
}