#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6


__kernel void sum(__global const unsigned int *a, __local unsigned int *temp, const unsigned int n, __global unsigned int *sum)
{
    const unsigned int iGroup = get_local_id(0);
    const unsigned int groupSize = get_local_size(0);
    const unsigned int groupCount = get_num_groups(0);
    
    const unsigned int iWarp = iGroup % WARP_SIZE;
    const unsigned int warpIndex = iGroup / WARP_SIZE;
    const unsigned int warpCount = (groupSize + WARP_SIZE - 1) / WARP_SIZE;

    // setup
    if (get_global_id(0) == 0)
        *sum = 0;
    
    temp[iGroup] = a[get_global_id(0)];
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // variant 1: each warp adds elements from the next one in binary tree; the last remaining warp is summed manually 
    {
        for (unsigned int step = WARP_SIZE; step < warpCount * WARP_SIZE; step <<= 1) {
            const unsigned int target = 2 * warpIndex * step + iWarp;
            if (target + step < groupSize)
                temp[target] += temp[target + step];

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // variant 1.1: last warp sums itself as a binary tree
        {
            if (warpIndex == 0)
                for (unsigned int step = 0; (1 << step) < WARP_SIZE; ++step)
                    if (((2 * iWarp + 1) << step) < groupSize)
                        temp[iWarp << (step + 1)] += temp[(2 * iWarp + 1) << step];
                    // no barrier needed as we are operating within a single warp
            // again, no barrier, because temp[0] will be read by a member of the same first warp

            if (iGroup == 0)
                atomic_add(sum, temp[0]);
        }

        // variant 1.2: last warp atomically adds its elements to the sum
        // {
        //     if (warpIndex == 0)
        //         atomic_add(sum, temp[iWarp]);
        // }
    }

    // variant 2: sums are calculated within each warp and then summed using binary tree order
    // {
    //     for (unsigned int step = 1; step < WARP_SIZE; step <<= 1)
    //         if (iWarp % (2 * step) == 0 && iGroup + step < groupSize)
    //             temp[iGroup] += temp[iGroup + step];
    //         // no barrier needed as we are operating within a single warp
    //     barrier(CLK_LOCAL_MEM_FENCE); 

    //     for (unsigned int step = 1; step < warpCount; step <<= 1) {
    //         if (warpIndex % (2 * step) == 0 && iGroup + step * WARP_SIZE < groupSize)
    //             temp[iGroup] += temp[iGroup + step * WARP_SIZE];

    //         barrier(CLK_LOCAL_MEM_FENCE);
    //     }

    //     if (iGroup == 0) {
    //         atomic_add(sum, temp[0]);
    //     }
    // }
}