#define SIZE 128
__kernel void max_prefix_sum(__global const int* sum,
                             __global const int* max_sum,
                             __global const int* index,
                             unsigned int n,
                             __global int* new_sum,
                             __global int* new_max_sum,
                             __global int* new_index) {
    __local int local_sum[SIZE];
    __local int local_max_sum[SIZE];
    __local int local_index[SIZE];

    const size_t group_id = get_group_id(0);
    const size_t local_id = get_local_id(0);
    const size_t global_id = group_id * get_local_size(0) + local_id;
    if (global_id < n) {
        local_sum[local_id] = sum[global_id];
        local_max_sum[local_id] = max_sum[global_id];
        local_index[local_id] = index[global_id];
    } else {
        local_sum[local_id] = 0;
        local_max_sum[local_id] = 0;
        local_index[local_id] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id == 0) {
        int tmp_sum = 0;
        int tmp_max_sum = 0;
        int tmp_index = 0;

        for (size_t i = 0; i < SIZE; ++i) {
            if (tmp_sum + local_max_sum[i] > tmp_max_sum) {
                tmp_max_sum = tmp_sum + local_max_sum[i];
                tmp_index = local_index[i];
            }
            tmp_sum += local_sum[i];
        }

        new_sum[group_id] = tmp_sum;
        new_max_sum[group_id] = tmp_max_sum;
        new_index[group_id] = tmp_index;
    }
}