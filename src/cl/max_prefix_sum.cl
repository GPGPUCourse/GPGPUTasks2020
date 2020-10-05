#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6
#define WORK_GROUP_SIZE 256

__kernel void max_prefix_sum(__global const int* from_sum,
                             __global const int* from_max,
                             __global const int* from_id,
                             __global       int* to_sum,
                             __global       int* to_max,
                             __global       int* to_id,
                                      const unsigned int n)
{
    const unsigned int global_index = get_global_id(0);
    const unsigned int local_index = get_local_id(0);
    const unsigned int group_index = get_group_id(0);

    // заведём long типы, чтобы при суммировании манимального
    // значения инта и отрицательного числа не было переполнения:
    __local long local_sum[WORK_GROUP_SIZE];
    __local long local_max[WORK_GROUP_SIZE];
    __local long local_id[WORK_GROUP_SIZE];
    if (global_index < n) {
        local_sum[local_index] = from_sum[global_index];
        local_max[local_index] = from_max[global_index];
        local_id[local_index] = from_id[global_index];
    } else {
        local_sum[local_index] = 0;
        local_max[local_index] = INT_MIN;
        local_id[local_index] = 0;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    for (int step = 1; step < WORK_GROUP_SIZE; step *= 2) {
        if (local_index % (step * 2) == 0) {
            if (local_max[local_index] < local_sum[local_index] + local_max[local_index + step]) {
                local_max[local_index] = max(local_max[local_index], local_sum[local_index] + local_max[local_index + step]);
                local_id[local_index] = local_id[local_index + step];
            }
            local_sum[local_index] += local_sum[local_index + step];
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    if (local_index == 0) {
        to_sum[group_index] = local_sum[0];
        to_max[group_index] = local_max[0];
        to_id[group_index] = local_id[0];
    }
}