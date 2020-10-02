#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6
#define WORK_GROUP_SIZE 256

int mymax (int a, int b) {
    if (a >= b) {
        return a;
    } else {
        return b;
    }
}

__kernel void max_prefix_sum(__global const unsigned int* from_sum,
                             __global const unsigned int* from_max,
                             __global       unsigned int* to_sum,
                             __global       unsigned int* to_max,
                                      const unsigned int n)
{
    const unsigned int global_index = get_global_id(0);
    const unsigned int local_index = get_local_id(0);
    const unsigned int group_index = get_group_id(0);

    __local unsigned int local_sum[WORK_GROUP_SIZE];
    __local unsigned int local_max[WORK_GROUP_SIZE];
    if (global_index < n) {
        local_sum[local_index] = from_sum[global_index];
        local_max[local_index] = from_max[global_index];
    } else {
        local_sum[local_index] = 0;
        local_max[local_index] = INT_MIN;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int step = 1; step < WORK_GROUP_SIZE; step *= 2) {
        if (local_index % (step * 2) == 0) {
            local_max[local_index] = mymax(local_max[local_index], local_sum[local_index] + local_max[local_index + step]);
            local_sum[local_index] += local_sum[local_index + step];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_index == 0) {
        to_sum[group_index] = local_sum[0];
        to_max[group_index] = local_max[0];
    }
}