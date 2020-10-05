#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6
#define WORK_GROUP_SIZE 128
#define DOUBLE_WORK_GROUP_SIZE 256

__kernel void prefix_k(__global int* array, __global int* buffer, unsigned int n) {
    int local_id = get_local_id(0);
    int global_id = get_global_id(0);

    __local int local_mem[WORK_GROUP_SIZE];
    __local int tree[DOUBLE_WORK_GROUP_SIZE];

    if (global_id < n) {
        local_mem[local_id] = array[global_id];
        tree[local_id + WORK_GROUP_SIZE] = local_mem[local_id];
    } else {
        local_mem[local_id] = 0;
        tree[local_id + WORK_GROUP_SIZE] = 0;
    }
    tree[local_id] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);
    for (int step = WORK_GROUP_SIZE / 2; step > 0; step /= 2) {
        if (local_id < step) {
            int ch_id = (local_id + step) * 2;
            tree[local_id + step] = tree[ch_id] + tree[ch_id + 1];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        tree[1] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int step = 1; step < WORK_GROUP_SIZE; step *= 2) {
        int ch_id = (local_id + step) * 2;
        if (local_id < step) {
            tree[ch_id + 1] = tree[ch_id] + tree[local_id + step];
            tree[ch_id] = tree[local_id + step];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == WORK_GROUP_SIZE - 1) {
        buffer[get_group_id(0)] = local_mem[local_id] + tree[local_id + WORK_GROUP_SIZE];
    }
    array[global_id] = local_mem[local_id] + tree[local_id + WORK_GROUP_SIZE];
}

__kernel void sum_k(__global int* array, __global int* buffer, unsigned int n) {
    int global_id = get_global_id(0);
    int group_id = get_group_id(0);

    if (global_id < n && group_id > 0) {
        array[global_id] += buffer[group_id - 1];
    }
}