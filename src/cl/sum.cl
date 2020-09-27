#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 128
__kernel void sum(__global const unsigned int* a,
                  __global unsigned int* sum,
                  unsigned int n)
{
    const unsigned int index = get_global_id(0);
    const unsigned int local_index = get_local_id(0);

    __local int local_a[WORK_GROUP_SIZE];
    if (index >= n) {
        local_a[local_index] = 0;
    } else {
        local_a[local_index] = a[index];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    int local_sum = 0;
    for (int i = WARP_SIZE; i > 1; i /= 2) {
        if ((local_index % WARP_SIZE) * 2 < i) {
            local_a[local_index] += local_a[local_index + i/2];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_index == 0) {
        for (int i = 0; i < WORK_GROUP_SIZE; i += WARP_SIZE) {
            local_sum += local_a[i];
        }
        atomic_add(sum, local_sum);
    }
}