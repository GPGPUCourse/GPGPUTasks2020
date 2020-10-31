#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void max_prefix_sum(__global const int* sm,
                             __global const int* px,
                             __global const int* idx,
                             unsigned int n,
                             __global int* sm_out,
                             __global int* px_out,
                             __global int* idx_out)
{
    int local_id = get_local_id(0);
    int global_id = get_global_id(0);
    int group_id = get_group_id(0);

    __local int local_sm[WORK_GROUP_SIZE];
    __local int local_px[WORK_GROUP_SIZE];
    __local int local_idx[WORK_GROUP_SIZE];

    if (global_id < n) {
        local_sm[local_id] = sm[global_id];
        local_px[local_id] = px[global_id];
        local_idx[local_id] = idx[global_id];
    } else {
        local_sm[local_id] = 0;
        local_px[local_id] = 0;
        local_idx[local_id] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id == 0) {
        int max_sum = 0;
        int sum = 0;
        int result = 0;
        for (int i = 0; i < WORK_GROUP_SIZE; ++i) {
            if (max_sum < sum + local_px[i]) {
                max_sum = sum + local_px[i];
                result = local_idx[i];
            }
            sum += local_sm[i];
        }
        sm_out[group_id] = sum;
        px_out[group_id] = max_sum;
        idx_out[group_id] = result;
    }
}