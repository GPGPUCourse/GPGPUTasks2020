#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6
#ifndef WORK_GROUP_SIZE
    #define WORK_GROUP_SIZE 128
#endif
__kernel void max_prefix_sum(__global const int *max_pref, __global const int *sum, __global const unsigned int *index_range,
                             __global int *res_max_pref, __global int *res_sum, __global unsigned int *res_index_range,
                             const unsigned int n)
{
    const unsigned int global_id = get_global_id(0);
    const unsigned int group_id = get_group_id(0);
    const unsigned int local_id = get_local_id(0);

    __local int local_max_pref[WORK_GROUP_SIZE];
    __local int local_sum[WORK_GROUP_SIZE];
    __local int local_index_range[WORK_GROUP_SIZE];

    if (global_id < n) {
        local_max_pref[local_id] = max_pref[global_id];
        local_sum[local_id] = sum[global_id];
        local_index_range[local_id] = index_range[global_id];
    } else {
        local_max_pref[local_id] = 0;
        local_sum[local_id] = 0;
        local_index_range[local_id] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // simple version w/ summation via one thread
    // TO DO: Update to multiple thread version
    if (local_id == 0){
        int current_pref_sum = 0;
        int current_sum = 0;
        unsigned int current_index = 0;
        for (unsigned int i = 0; i < WORK_GROUP_SIZE; ++i) {
            // trying to merge current maximum prefix with another block
            if (current_pref_sum < local_max_pref[i] + current_sum) {
                current_pref_sum = local_max_pref[i] + current_sum;
                current_index = local_index_range[i];
            }
            current_sum += local_sum[i];
        }

        res_max_pref[group_id] = current_pref_sum;
        res_sum[group_id] = current_sum;
        res_index_range[group_id] = current_index;
    }
}


