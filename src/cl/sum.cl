#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define VALS_IN_STEP 16
#define WORKGROUP_SIZE 128

__kernel void sum(
    __global unsigned int* from,
    __global unsigned int* to,
    unsigned int n)
{
    size_t localId = get_local_id(0);
    size_t globalId = get_global_id(0);

    int toIdx = globalId;
    int fromIdxStart = toIdx/VALS_IN_STEP/WORKGROUP_SIZE*VALS_IN_STEP*WORKGROUP_SIZE;

    __local unsigned int local_mem[VALS_IN_STEP*WORKGROUP_SIZE];
    /*
    local_mem[localId] = from[globalId];
    local_mem[localId + 16] = from[globalId + 16];
    local_mem[localId + 32] = from[globalId + 32];
    local_mem[localId] = from[globalId];
    local_mem[localId] = from[globalId];
    local_mem[localId] = from[globalId];
    */
    // printf("%d + %d*[0, %d)\n", fromIdxStart, WORKGROUP_SIZE, VALS_IN_STEP);
    for (int i = 0; i < VALS_IN_STEP; ++i) {
        unsigned int idx = globalId/WORKGROUP_SIZE*WORKGROUP_SIZE*VALS_IN_STEP + i*WORKGROUP_SIZE + localId;
        if (idx < n) {
            printf("from[%d] = %d\n", idx, from[idx]);
            local_mem[i*WORKGROUP_SIZE + localId] = from[idx];
        } else {
            local_mem[i*WORKGROUP_SIZE + localId] = 0;
        }
    }
    unsigned int sum = 0;
    for (int i = 0; i < VALS_IN_STEP; ++i) {
        sum += local_mem[i*WORKGROUP_SIZE + localId];
    }
    printf("to[%d] = %d\n", toIdx, sum);
    to[toIdx] = sum;
}
