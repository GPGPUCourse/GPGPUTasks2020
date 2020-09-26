#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

typedef unsigned int index_t;
 
#define joinSegments(sumLeft, bestSumLeft, bestIndexLeft, sumRight, bestSumRight, bestIndexRight) \
    if (bestIndexLeft != 0 && bestSumLeft + sumRight > bestSumRight) {                           \
        bestSumLeft += sumRight;                                                                  \
    } else {                                                                                      \
        bestSumLeft   = bestSumRight;                                                             \
        bestIndexLeft = bestIndexRight;                                                           \
    }                                                                                             \
    sumLeft += sumRight

__kernel void max_prefix_sum(
    __global const int *a, const index_t n,                                                    // input
     __local int *sumLocal, __local index_t *bestPrefixLocal, __local int *bestPrefixSumLocal, // storage for results, computed within work groups 
    __global int *sum,     __global index_t *bestPrefix,     __global int *bestPrefixSum       // storage for merging work group results
) {                                                                                            // data is separated to distinct arrays for better cache coalescence
    const index_t iGlobal = get_global_id(0);
    const index_t globalSize = get_global_size(0);

    const index_t iGroup = get_local_id(0);
    const index_t groupSize = get_local_size(0);
    const index_t groupCount = get_num_groups(0);
    const index_t groupIndex = get_group_id(0);
    
    const index_t iWarp = iGroup % WARP_SIZE;
    const index_t warpIndex = iGroup / WARP_SIZE;
    const index_t warpCount = (groupSize + WARP_SIZE - 1) / WARP_SIZE;

    // setup work group stage
    {
        const index_t reversedIndex = globalSize - iGlobal - 1;
        if (reversedIndex < n) {
            sumLocal[iGroup] = a[reversedIndex];
            bestPrefixLocal[iGroup] = reversedIndex + 1;
        } else {
            sumLocal[iGroup] = 0;
            bestPrefixLocal[iGroup] = 0;
        }
        bestPrefixSumLocal[iGroup] = sumLocal[iGroup];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // compute values inside each warp
    for (index_t step = 1; step < WARP_SIZE; step <<= 1) {
        const index_t inWarpTarget = 2 * iWarp * step;
        const index_t target = warpIndex * WARP_SIZE + inWarpTarget;
        const index_t source = target + step;
        
        if (inWarpTarget + step < WARP_SIZE && source < groupSize) {
            joinSegments(
                sumLocal[target], bestPrefixSumLocal[target], bestPrefixLocal[target],
                sumLocal[source], bestPrefixSumLocal[source], bestPrefixLocal[source]
            );
        }
        // no barrier needed as we are operating within warps
    }
    barrier(CLK_LOCAL_MEM_FENCE); // synchronize

    // join warp results pretty much the same way
    for (index_t step = WARP_SIZE; step < warpCount * WARP_SIZE; step <<= 1) {
        const index_t target = 2 * iGroup * step;
        const index_t source = target + step;
        
        if (source < groupSize) {
            joinSegments(
                sumLocal[target], bestPrefixSumLocal[target], bestPrefixLocal[target],
                sumLocal[source], bestPrefixSumLocal[source], bestPrefixLocal[source]
            );
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // flush results from each group to global memory
    if (iGroup == 0) {
        sum          [groupIndex] = sumLocal          [0];
        bestPrefixSum[groupIndex] = bestPrefixSumLocal[0];
        bestPrefix   [groupIndex] = bestPrefixLocal   [0];
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    // join group results exactly the same way as before
    for (index_t step = 1; step < groupCount; step <<= 1) {
        const index_t target = 2 * iGlobal * step;
        const index_t source = target + step;

        if (source < groupCount) {
            joinSegments(
                sum[target], bestPrefixSum[target], bestPrefix[target],
                sum[source], bestPrefixSum[source], bestPrefix[source]
            );
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}