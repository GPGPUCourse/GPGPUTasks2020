#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 128
__kernel void max_prefix_sum(__global const int* a,
                             __global const int* prefix_sum,
                             __global int* sum,
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

        int bst_sum = local_sum;
        int bst_idx = 0;
        for (int i = 0; i < WORK_GROUP_SIZE; ++i) {
            local_sum += local_a[i];
            if (local_sum > bst_sum) {
                bst_sum = local_sum;
                bst_idx = i + 1;
            }
        }

        bst_idx += (index - local_index);
        atomic_max(sum, bst_sum);

        int prev_val = res[0];
        while (sum[0] == bst_sum && bst_idx != prev_val) {
            atomic_cmpxchg(res, prev_val, bst_idx);
            prev_val = res[0];
        }
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
    if (local_index == 0) {
        sum[bucket_id] = 0;
        for (int i = 0; i < WORK_GROUP_SIZE; ++i) {
            sum[bucket_id] += local_a[i];
        }
    }
}