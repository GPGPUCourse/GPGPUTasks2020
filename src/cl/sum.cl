__kernel void sum(__global const unsigned int *a, 
		__local unsigned int *local_sum, 
		unsigned int n, 
		__global unsigned int *sum)
{
    const unsigned int global_id = get_global_id(0);
    const unsigned int local_id = get_local_id(0);
    const unsigned int group_id = get_group_id(0);
    const unsigned int group_size = get_local_size(0);
    if (global_id < n) {
        local_sum[local_id] = a[global_id];
    } else {
        local_sum[local_id] = 0;
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
   
    for (unsigned int shift = group_size / 2; shift > 0; shift /= 2) {
        if (local_id < shift) {
            local_sum[local_id] += local_sum[local_id + shift];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        sum[group_id] = local_sum[0];
    }
}