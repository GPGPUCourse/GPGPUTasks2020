#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 128
__kernel void max_prefix_sum(__global const int* a,
                             __global const int* prefix_sum,
                             __global int* sum,
                             unsigned int n)
{
    const unsigned int index = get_global_id(0);
    const unsigned int bucket_id = index/WORK_GROUP_SIZE;
    const unsigned int local_index = get_local_id(0);

    __local int local_a[WORK_GROUP_SIZE];

    if (index >= n) {
        local_a[local_index] = 0;
    } else {
        local_a[local_index] = a[index];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_index == 0) {
        int local_sum = prefix_sum[bucket_id];

        int bst_sum = local_sum;
        for (int i = 0; i < WORK_GROUP_SIZE; ++i) {
            local_sum += local_a[i];
            if (local_sum > bst_sum) {
                bst_sum = local_sum;
            }
        }

        atomic_max(sum, bst_sum);
    }
}

__kernel void max_prefix_sum_index(__global const int* a,
                                   __global const int* prefix_sum,
                                   const int sum,
                                   __global int* res,
                                   unsigned int n)
{
    const unsigned int index = get_global_id(0);
    const unsigned int bucket_id = index/WORK_GROUP_SIZE;
    const unsigned int local_index = get_local_id(0);

    __local int local_a[WORK_GROUP_SIZE];

    if (index >= n) {
        local_a[local_index] = 0;
    } else {
        local_a[local_index] = a[index];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_index == 0) {
        int local_sum = prefix_sum[bucket_id];

        int bst_idx = -1;
        if (local_sum == sum) {
            bst_idx = 0;
        } else {
            for (int i = 0; i < WORK_GROUP_SIZE; ++i) {
                local_sum += local_a[i];
                if (local_sum == sum) {
                    bst_idx = i + 1;
                    break;
                }
            }
        }

        if (bst_idx != -1) {
            bst_idx += (index - local_index);
            atomic_min(res, bst_idx);
        }
    }
}

__kernel void calc_prefix_sum(__global const int* a,
                              __global const int* prefix_sum,
                              __global int* sum,
                              unsigned int n)
{
    const unsigned int index = get_global_id(0);
    const unsigned int bucket_id = index/WORK_GROUP_SIZE;
    const unsigned int local_index = get_local_id(0);

    __local int local_a[WORK_GROUP_SIZE];

    if (index >= n) {
        local_a[local_index] = 0;
    } else {
        local_a[local_index] = a[index];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_index == 0) {
        int local_sum = prefix_sum[bucket_id];

        for (int i = 0; i < WORK_GROUP_SIZE; ++i) {
            local_sum += local_a[i];
            local_a[i] = local_sum;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (index < n) {
        sum[index] = local_a[local_index];
    }
}


__kernel void sum_in_bucket(__global const int* a,
                            __global int* sum,
                            unsigned int n)
{
    const unsigned int index = get_global_id(0);
    const unsigned int bucket_id = index/WORK_GROUP_SIZE;
    const unsigned int local_index = get_local_id(0);

    __local int local_a[WORK_GROUP_SIZE];

    if (index >= n) {
        local_a[local_index] = 0;
    } else {
        local_a[local_index] = a[index];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    sum[bucket_id] = 0;
    for (int i = WARP_SIZE; i > 1; i /= 2) {
        if ((local_index % WARP_SIZE) * 2 < i) {
            local_a[local_index] += local_a[local_index + i/2];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_index == 0) {
         for (int i = WARP_SIZE; i < WORK_GROUP_SIZE; i += WARP_SIZE) {
             local_a[0] += local_a[i];
         }
         sum[bucket_id]  = local_a[0];
    }
}