#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6
#define WORK_GROUP_SIZE 128

__kernel void sum_k(__global const unsigned int* array, __global unsigned int* res, unsigned int n) {
    int local_id = get_local_id(0);
    int global_id = get_global_id(0);

    __local unsigned int local_mem[WORK_GROUP_SIZE];
    if (global_id < n) {
        local_mem[local_id] = array[global_id];
    } else {
        local_mem[local_id] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    for (int current_n = WORK_GROUP_SIZE; current_n > 1; current_n /= 2) {
        if (2 * local_id < current_n) {
            local_mem[local_id] += local_mem[local_id + current_n / 2];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        atomic_add(res, local_mem[local_id]);
    }
}