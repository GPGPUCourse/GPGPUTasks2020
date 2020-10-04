#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#define WORK_GROUP_SIZE 128

#line 8

__kernel void max_prefix_sum(__global const int* sm,
                             __global const int* px,
                             unsigned int n,
                             __global int* sm_out,
                             __global int* px_out)
{
    int local_id = get_local_id(0);
    int global_id = get_global_id(0);

    __local int local_sm[WORK_GROUP_SIZE];
    __local int local_px[WORK_GROUP_SIZE];
    if (global_id >= n) {
        local_sm[local_id] = 0;
        local_px[local_id] = 0;
    } else {
        local_sm[local_id] = sm[global_id];
        local_px[local_id] = px[global_id];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 0; i < WORK_GROUP_SIZE; i += 2) {
//        printf("local_id=%i i=%i local_px[i]=%i local_sm[i] + local_px[i + 1]=%i\n", local_id, i, local_px[i], local_sm[i] + local_px[i + 1]);
        local_px[i] = max(local_px[i], local_sm[i] + local_px[i + 1]);
        local_sm[i] = local_sm[i] + local_sm[i + 1];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_id % 2 == 0) {
        sm_out[local_id / 2] = local_sm[local_id];
        px_out[local_id / 2] = local_px[local_id];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}
