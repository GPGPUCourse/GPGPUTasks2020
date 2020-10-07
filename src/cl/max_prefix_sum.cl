#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 8

__kernel void
maxPrefixSum(__global const int *gSum, __global const int *gMaxSum, __global const uint *gResult, __global int *gSumW,
             __global int *gMaxSumW, __global uint *gResultW, uint n) {
    const uint globalId = get_global_id(0);
    const uint localId = get_local_id(0);
    const uint groupId = get_group_id(0);
    const uint localSize = 128; // get_local_size(0)
    __local int lSum[localSize];
    __local int lMaxSum[localSize];
    __local int lResult[localSize];
    if (globalId < n) {
        lSum[localId] = gSum[globalId];
        lMaxSum[localId] = gMaxSum[globalId];
        lResult[localId] = gResult[globalId];
    } else {
        lSum[localId] = 0;
        lMaxSum[localId] = 0;
        lResult[localId] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (localId == 0) {
        int sum = 0;
        int maxSum = -2147483648;
        uint result = 0;
        for (uint i = 0; i != localSize; ++i) {
            int temp = sum + lMaxSum[i];
            if (temp > maxSum) {
                maxSum = temp;
                result = lResult[i];
            }
            sum += lSum[i];
        }
        gSumW[groupId] = sum;
        gMaxSumW[groupId] = maxSum;
        gResultW[groupId] = result;
    }
}