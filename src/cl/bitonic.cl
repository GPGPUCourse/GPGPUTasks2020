#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define GROUP_SIZE 256

void local_sort(__local float* local_as, unsigned int size) {
    unsigned int local_id = get_local_id(0);

    unsigned int other_id = local_id + size - 1 - local_id % size;
    if (local_id % size < size / 2 && local_as[local_id] > local_as[other_id]) {
        float tmp = local_as[local_id];
        local_as[local_id] = local_as[other_id];
        local_as[other_id] = tmp;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int i = size / 2; i > 1; i /= 2) {
        other_id = local_id + i / 2;
        if (local_id % i < i / 2 && local_as[local_id] > local_as[other_id]) {
            float tmp = local_as[local_id];
            local_as[local_id] = local_as[other_id];
            local_as[other_id] = tmp;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

void global_sort(__global float* global_as, unsigned int size, unsigned int first) {
    unsigned int global_id = get_global_id(0);

    unsigned int other_id = 0;
    if (global_id % size < size / 2) {
        if (first != 0) {
            other_id = global_id + size - 1 - global_id % size;
        } else {
            other_id = global_id + size / 2;
        }

        if (global_as[global_id] > global_as[other_id]) {
            float tmp = global_as[global_id];
            global_as[global_id] = global_as[other_id];
            global_as[other_id] = tmp;
        }
    }
}

__kernel void bitonic(__global float* as, unsigned int size, unsigned int first) {
    unsigned int local_id = get_local_id(0);
    unsigned int global_id = get_global_id(0);
    if (size <= GROUP_SIZE) {
        __local float local_as[GROUP_SIZE];
        local_as[local_id] = as[global_id];
        if (size == GROUP_SIZE) {
            local_sort(local_as, size);
        } else {
            for (int i = 2; i <= GROUP_SIZE; i *= 2) {
                local_sort(local_as, i);
            }
        }
        as[global_id] = local_as[local_id];
    } else {
        global_sort(as, size, first);
    }
}
