#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 8

#ifndef WORK_GROUP_SIZE
#define WORK_GROUP_SIZE 128
#endif

__kernel void bitonic_global(__global float *as,
                             unsigned int n,
                             unsigned int box_size,
                             unsigned int distance) {
    int global_id = get_global_id(0);
    bool should_be_ordered = global_id % (2 * box_size) < box_size;

    int id = (global_id / distance) * 2 * distance + global_id % distance;
    if (id + distance < n) {
        float num = as[id];
        if ((as[id + distance] < num) == should_be_ordered) {
            as[id] = as[id + distance];
            as[id + distance] = num;
        }
    }
}

__kernel void bitonic_local(__global float *as,
                            unsigned int n,
                            unsigned int box_size,
                            unsigned int distance) {
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);

    __local float local_sample[2 * WORK_GROUP_SIZE];

    if (2 * global_id + 1 < n) {
        local_sample[2 * local_id] = as[2 * global_id];
        local_sample[2 * local_id + 1] = as[2 * global_id + 1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    bool should_be_ordered = global_id % (2 * box_size) < box_size;

    for (int dist = distance; 1 <= dist; dist >>= 1) {
        int id = (local_id / dist) * 2 * dist + local_id % dist;

        if (global_id - local_id + id + dist < n) {
            float num = local_sample[id];
            if ((local_sample[id + dist] < num) == should_be_ordered) {
                local_sample[id] = local_sample[id + dist];
                local_sample[id + dist] = num;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (2 * global_id + 1 < n) {
        as[2 * global_id] = local_sample[2 * local_id];
        as[2 * global_id + 1] = local_sample[2 * local_id + 1];
    }
}