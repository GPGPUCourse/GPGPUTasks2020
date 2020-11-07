#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 128

__kernel void count_prefix(__global int* as, __global int* buffer, unsigned int n) {
    int local_id = get_local_id(0);
    int global_id = get_global_id(0);
    int group_id = get_group_id(0);

    __local int local_as[WORK_GROUP_SIZE];
    __local int tree[WORK_GROUP_SIZE * 2];

    if (global_id < n) {
        local_as[local_id] = as[global_id];
        tree[local_id + WORK_GROUP_SIZE] = local_as[local_id];
    } else {
        local_as[local_id] = 0;
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
        buffer[group_id] = local_as[local_id] + tree[local_id + WORK_GROUP_SIZE];
    }
    as[global_id] = local_as[local_id] + tree[local_id + WORK_GROUP_SIZE];
}

__kernel void count_sum(__global int* array, __global int* buffer, unsigned int n) {
    int global_id = get_global_id(0);
    int group_id = get_group_id(0);

    if (global_id < n && group_id > 0) {
        array[global_id] += buffer[group_id - 1];
    }
}

__kernel void radix_bits(__global unsigned int* as, __global int* zeroes, __global int* ones, unsigned int nth_bit, unsigned int n) {
    int global_id = get_global_id(0);
    if (global_id < n) {
        unsigned int number = as[global_id];
        int bit_value = 1 == ((number >> nth_bit) & 1);
        if (bit_value) {
            ones[global_id] = 1;
            zeroes[global_id] = 0;
        } else {
            zeroes[global_id] = 1;
            ones[global_id] = 0;
        }
    }
}

__kernel void radix_set(__global unsigned int* as_old, __global unsigned int* as_new, __global int* zeroes, __global int* ones,
                        unsigned int nth_bit, unsigned int n) {
    int global_id = get_global_id(0);
    if (global_id < n) {
        unsigned int number = as_old[global_id];
        int bit_value = 1 == ((number >> nth_bit) & 1);
        if (bit_value) {
            as_new[ones[global_id] + zeroes[n - 1] - 1] = number;
        } else {
            as_new[zeroes[global_id] - 1] = number;
        }
    }
}