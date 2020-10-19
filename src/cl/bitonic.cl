#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void bitonic(__global float* as, int N, int BLOCK_SIZE, int GLOBAL_BLOCK_SIZE) {
    const unsigned int id = get_global_id(0);
    if (id % (2*BLOCK_SIZE) < BLOCK_SIZE && id < N) {
           const unsigned int next_id = id + BLOCK_SIZE;
           if (next_id < N) {
               if ((as[id] > as[next_id] && id % (4 * GLOBAL_BLOCK_SIZE) < 2 * GLOBAL_BLOCK_SIZE) ||
                   (as[id] < as[next_id] && id % (4 * GLOBAL_BLOCK_SIZE) >= 2 * GLOBAL_BLOCK_SIZE)) {
                   float tmp = as[id];
                   as[id] = as[next_id];
                   as[next_id] = tmp;
               }
           }
    }
}

#define WORK_GROUP_SIZE 128
__kernel void sort_block(__global float* as, int N, int GLOBAL_BLOCK_SIZE) {
    const unsigned int id = get_global_id(0);
    const unsigned int local_id = get_local_id(0);

    __local float local_a[WORK_GROUP_SIZE];
    int order = 1;
    if (id % (4 * GLOBAL_BLOCK_SIZE) >= 2 * GLOBAL_BLOCK_SIZE) {
        order = -1;
    }

    if (id < N) {
        local_a[local_id] = as[id];
    } else {
        local_a[local_id] = 1e9 * order;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int BLOCK_SIZE = WORK_GROUP_SIZE/2;
    if (GLOBAL_BLOCK_SIZE < BLOCK_SIZE) {
        BLOCK_SIZE = GLOBAL_BLOCK_SIZE;
    }

    for (; BLOCK_SIZE >= 1; BLOCK_SIZE /= 2) {
        if (local_id % (2*BLOCK_SIZE) < BLOCK_SIZE) {
               const unsigned int next_id = local_id + BLOCK_SIZE;
               if ((local_a[local_id] > local_a[next_id] && order == 1) ||
                   (local_a[local_id] < local_a[next_id] && order == -1)) {
                  float tmp = local_a[local_id];
                  local_a[local_id] = local_a[next_id];
                  local_a[next_id] = tmp;
               }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    as[id] = local_a[local_id];
}
