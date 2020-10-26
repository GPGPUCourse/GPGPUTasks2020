__kernel void radix_setup(
    const __global unsigned int *as, 
    const unsigned int n, 
    __global unsigned int *counts,
    const unsigned int bit
) {
    const size_t iGlobal = get_global_id(0);
    const size_t iGroup = get_local_id(0);
    
    if (iGlobal == 0) {
        counts[0] = 0;
    }

    if (iGlobal < n) {
        counts[iGlobal + 1] = (as[iGlobal] >> bit) & 1;
    }
}

__kernel void radix_gather(
    __global unsigned int *as, 
    const unsigned int n,
    const unsigned int globalStep
) {
    const size_t iGlobal = get_global_id(0) * globalStep;
    const size_t iGroup = get_local_id(0);
    
    __local unsigned int asLocal[LOCAL_SIZE];
   
    if (iGlobal < n) {
        asLocal[iGroup] = as[iGlobal + 1];
    } else {
        asLocal[iGroup] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (size_t step = 1; step < LOCAL_SIZE; step <<= 1) {
        if (iGroup >= step) {
            asLocal[iGroup] += asLocal[iGroup - step];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (iGlobal < n) {
        as[iGlobal + 1] = asLocal[iGroup];
    }
}

__kernel void radix_propagate(
    __global unsigned int *as, 
    const unsigned int n
) {
    const size_t iGlobal = get_global_id(0);
    const size_t iGroup = get_local_id(0);

    if (iGroup > 0) {
        as[iGlobal + 1] += as[iGlobal + 1 - iGroup];
    }
}

__kernel void radix_move(
    const __global unsigned int *as, 
    const unsigned int n, 
    const __global unsigned int *counts, 
    __global unsigned int *bs
) {
    const size_t iGroup = get_group_id(0);
    const size_t iGlobal = get_global_id(0);

    const size_t zeros = n - counts[n];

    if (iGlobal < n) {
        const size_t index = 
            counts[iGlobal + 1] == counts[iGlobal]
            ? iGlobal - counts[iGlobal]
            : zeros + counts[iGlobal];
            
        bs[index] = as[iGlobal];
    }

    // const size_t ones = counts[n];
    // const size_t zeros = n - ones;

    // const size_t isZero = iGlobal + ones < n;
    // const size_t indexToSearch = isZero ? iGlobal - counts[iGlobal] : zeros + counts[iGlobal];
    
    // size_t l = 0;
    // size_t r = n + 1;
    // while (l + 1 < r) {
    //     const size_t m = (l + r) >> 1;
        
    //     size_t index;
    //     if (isZero) {
    //         index = m - counts[m];
    //     } else {
    //         index = zeros + counts[m];
    //     }

    //     if (index > indexToSearch) {
    //         r = m;
    //     } else {
    //         l = m;
    //     }
    // }
    // barrier(CLK_GLOBAL_MEM_FENCE); // just to synchronize everything

    // bs[iGlobal] = as[l - 1];
}
