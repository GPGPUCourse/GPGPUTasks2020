#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 128

__kernel void sum(__global const unsigned int* a, int n, __global unsigned int* res) {
    int local_index = get_local_id(0);
    int global_index = get_global_id(0);

    __local unsigned int local_a[WORK_GROUP_SIZE];

    if (global_index >= n) {
        local_a[local_index] = 0;
    }
    else {
        local_a[local_index] = a[global_index];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_index == 0) {
        unsigned int sum = 0;
        for (int i = 0; i < WORK_GROUP_SIZE; i++) {
            sum += local_a[i];
        }
        atomic_add(res, sum);
    }
}