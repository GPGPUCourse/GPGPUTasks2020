#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void sum(__global const unsigned int* array,
                  __global unsigned int* res,
                  unsigned int n)
{
    int local_id = get_local_id(0);
    int global_id = get_global_id(0);
    int group_id = get_group_id(0);

    __local int local_array[WORK_GROUP_SIZE];
    if (global_id < n) {
        local_array[local_id] = array[global_id];
    } else {
        local_array[local_id] = 0;
    }


    barrier(CLK_LOCAL_MEM_FENCE);
    for (int n_values = WORK_GROUP_SIZE; n_values > 1; n_values /= 2) {
        if (2 * local_id < n_values) {
            unsigned int a = local_array[local_id];
            unsigned int b = local_array[local_id + n_values / 2];
            local_array[local_id] = a + b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_id == 0) {
        res[group_id] = local_array[0];
    }
}

 __kernel void sum_tree(__global const int* array,
                   __global int* sum_array,
                   unsigned int n) {
    int local_id = get_local_id(0);
    int global_id = get_global_id(0);
    int group_id = get_group_id(0);

    __local int local_array[WORK_GROUP_SIZE];
    if (global_id < n) {
        local_array[local_id] = array[global_id];
    } else {
        local_array[local_id] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    int sum = 0;
    for (int n_values = WORK_GROUP_SIZE; n_values > 1; n_values /= 2) {
        if (local_id * 2 < n_values) {
            unsigned int a = local_array[local_id];
            unsigned int b = local_array[local_id + n_values / 2];
            local_array[local_id] = a + b;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_id == 0)
        sum_array[group_id] = local_array[0];
}
