#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6


__kernel void sum(__global const unsigned int *a, __local unsigned int *temp, const unsigned int n, __global unsigned int *sum)
{
    const unsigned int iGlobal = get_global_id(0);

    const unsigned int iGroup = get_local_id(0);
    const unsigned int groupIndex = get_group_id(0);
    const unsigned int groupSize = get_local_size(0);
    
    const unsigned int iWarp = iGroup % WARP_SIZE;
    const unsigned int warpIndex = iGroup / WARP_SIZE;
    const unsigned int warpCount = (groupSize + WARP_SIZE - 1) / WARP_SIZE;

    // setup
    temp[iGroup] = iGlobal < n ? a[iGlobal] : 0;
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    
    // sum warps
    for (unsigned int step = WARP_SIZE; step < warpCount * WARP_SIZE; step <<= 1) {
        const unsigned int target = 2 * warpIndex * step + iWarp;
        if (target + step < groupSize)
            temp[target] += temp[target + step];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // sum inside the first warp
    for (unsigned int step = 0; (step << 1) < WARP_SIZE; ++step) {
        const unsigned int target = 2 * iGroup << step;
        const unsigned int source = (2 * iGroup + 1) << step;
        if (source < WARP_SIZE)
            temp[target] += temp[source];
        // no barrier needed as we are operating within a single warp
    }
    // again, no barrier, because temp[0] will be read by a member of the same first warp

    if (iGroup == 0)
        sum[groupIndex] = temp[0];
}