#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#define VALS_IN_STEP 16
#define WORKGROUP_SIZE 128
#endif

#line 8

__kernel void sum(
    __global unsigned int* from,
    __global unsigned int* to,
    unsigned int n)
{
    unsigned int localId = get_local_id(0);
    unsigned int globalId = get_global_id(0);

    unsigned int toIdx = globalId;

    __local unsigned int local_mem[VALS_IN_STEP*WORKGROUP_SIZE];
    const unsigned int sumSize = (n + VALS_IN_STEP - 1)/VALS_IN_STEP;
    unsigned int sum = 0;
    for (int i = 0; i < VALS_IN_STEP; ++i) {
        unsigned int idx = sumSize*i + globalId/WORKGROUP_SIZE*WORKGROUP_SIZE + localId;
        if (idx < sumSize*(i+1) && idx < n) {
            sum += from[idx];
        }
    }
    to[toIdx] = sum;
}
