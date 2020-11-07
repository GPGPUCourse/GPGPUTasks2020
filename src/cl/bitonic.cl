#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define GROUP_SIZE 256

bool compare(float a, float b, bool mode) {
    return (mode) ? a > b : a < b;
}

void local_sort(__local float* as, unsigned int size, unsigned int chunk_size) {
    unsigned int local_id = get_local_id(0);
    unsigned int global_id = get_global_id(0);

    for (unsigned int i = size; i >= 2; i /= 2) {
        unsigned int other_id = local_id + i / 2;
        if (local_id % i < i / 2) {
            unsigned int mode = global_id % (2 * chunk_size) < chunk_size;
            if (compare(as[local_id], as[other_id], mode)) {
                float tmp = as[local_id];
                as[local_id] = as[other_id];
                as[other_id] = tmp;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

void global_sort(__global float* as, unsigned int size, unsigned int chunk_size) {
    unsigned int global_id = get_global_id(0);
    unsigned int other_id = global_id + size / 2;

    if (global_id % size < size / 2) {
        unsigned int mode = global_id % (2 * chunk_size) < chunk_size;
        if (compare(as[global_id], as[other_id], mode)) {
            float tmp = as[global_id];
            as[global_id] = as[other_id];
            as[other_id] = tmp;
        }
    }
}

__kernel void bitonic(__global float* as, unsigned int size, unsigned int chunk_size) {
    unsigned int local_id = get_local_id(0);
    unsigned int global_id = get_global_id(0);
    __local float local_as[GROUP_SIZE];

    if (size <= GROUP_SIZE) {
        local_as[local_id] = as[global_id];
        barrier(CLK_LOCAL_MEM_FENCE);

        if (size == GROUP_SIZE) {
            local_sort(local_as, size, chunk_size);
        } else {
            for (unsigned int i = 2; i <= GROUP_SIZE; i *= 2) {
                local_sort(local_as, i, i);
            }
        }

        as[global_id] = local_as[local_id];
    } else {
        global_sort(as, size, chunk_size);
    }
}
