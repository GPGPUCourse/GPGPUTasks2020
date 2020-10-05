__kernel void max_prefix_sum(
        __global const int *input_sum,
        __global const int *input_max_pref_sum,
        __global const unsigned int *input_max_pref_index,
		__local int *local_sum,
        __local int *local_max_pref_sum,
        __local unsigned int *local_max_pref_index,
		unsigned int n, 
		__global int *sum,
        __global int *max_pref_sum,
        __global unsigned int *max_pref_index)
{
    const unsigned int global_id = get_global_id(0);
    const unsigned int local_id = get_local_id(0);
    const unsigned int group_id = get_group_id(0);
    const unsigned int group_size = get_local_size(0);
    if (global_id < n) {
        local_sum[local_id] = input_sum[global_id];
        local_max_pref_sum[local_id] = input_max_pref_sum[global_id];
        local_max_pref_index[local_id] = input_max_pref_index[global_id];
    } else {
        local_sum[local_id] = 0;
        local_max_pref_sum[local_id] = 0;
        local_max_pref_index[local_id] = 0;
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
   
    for (unsigned int shift = group_size / 2; shift > 0; shift /= 2) {
        if (local_id < shift) {
            int left_pref_sum = local_max_pref_sum[2 * local_id];
            int right_pref_sum = local_max_pref_sum[2 * local_id + 1] + local_sum[2 * local_id];
            if (left_pref_sum >= right_pref_sum) {
                local_max_pref_sum[local_id] = left_pref_sum;
                local_max_pref_index[local_id] = local_max_pref_index[2 * local_id]; 
            } else {
                local_max_pref_sum[local_id] = right_pref_sum;
                local_max_pref_index[local_id] = local_max_pref_index[2 * local_id + 1]; 
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (local_id < shift) {
            local_sum[local_id] = local_sum[2 * local_id] + local_sum[2 * local_id + 1];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        sum[group_id] = local_sum[0];
        max_pref_sum[group_id] = local_max_pref_sum[0];
        max_pref_index[group_id] = local_max_pref_index[0];
    }
}