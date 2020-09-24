#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

typedef unsigned int index_t;
 
void joinSegments(
          int *sumLeft ,       int *bestSumLeft ,       index_t *bestIndexLeft,
    const int  sumRight, const int  bestSumRight, const index_t  bestIndexRight
) {
    if (*bestSumLeft +  sumRight >= bestIndexRight) { // as we reversed the array the minimal maximum sum prefix would be the rightmost one, hence >=
        *bestSumLeft += sumRight;
        // bestIndexLeft does not change
    } else {
        *bestSumLeft   = bestSumRight;
        *bestIndexLeft = bestIndexRight;
    }
    *sumLeft += sumRight;
}

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
    sumLocal[iGroup] = a[globalSize - iGlobal - 1];
    bestPrefixSumLocal[iGroup] = sumLocal[iGroup];
    bestPrefixLocal[iGroup] = globalSize - iGlobal - 1;
    barrier(CLK_LOCAL_MEM_FENCE);

    // compute values inside each warp
    for (index_t step = 0; (1 << step) < WARP_SIZE; ++step) {
        const index_t source = (2 * iWarp + 1) << step;
        const index_t target = iWarp << (step + 1);

        if (source < groupSize)
            joinSegments(
                sumLocal[target], bestPrefixSumLocal[target], bestPrefixLocal[target],
                sumLocal[source], bestPrefixSumLocal[source], bestPrefixLocal[source]
            );
        // no barrier needed as we are operating within a single warp
    }
    barrier(CLK_LOCAL_MEM_FENCE); // synchronize

    return;

    // join warp results pretty much the same way
    for (index_t step = WARP_SIZE; step < warpCount * WARP_SIZE; step <<= 1) {
        const index_t target = 2 * warpIndex * step;
        const index_t source = target + step;
        
        if (source < groupSize)
            joinSegments(
                sumLocal[target], bestPrefixSumLocal[target], bestPrefixLocal[target],
                sumLocal[source], bestPrefixSumLocal[source], bestPrefixLocal[source]
            );
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    return;
    
    // flush results from each group to global memory
    if (iGroup == 0) {
        sum          [groupIndex] = sumLocal          [0];
        bestPrefix   [groupIndex] = bestPrefixLocal   [0];
        bestPrefixSum[groupIndex] = bestPrefixSumLocal[0];
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    // join group results exactly the same way as before
    for (index_t step = 0; (1 << step) < groupCount; ++step) {
        const index_t source = (2 * iGlobal + 1) << step;
        const index_t target = iGlobal << (step + 1);

        if (source < groupCount)
            joinSegments(
                sum[target], bestPrefixSum[target], bestPrefix[target],
                sum[source], bestPrefixSum[source], bestPrefix[source]
            );
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}