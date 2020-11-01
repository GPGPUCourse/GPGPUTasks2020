__kernel void radix_setup(
    const __global unsigned int *as, 
    const unsigned int n, 
    __global unsigned int *a_cnts,
    const unsigned int bit
) {
    const size_t iGlobal = get_global_id(0);
    const size_t iGroup = get_local_id(0);
    
    if (iGlobal == 0) {
        a_cnts[0] = 0;
    }

    if (iGlobal < n) {
        a_cnts[iGlobal + 1] = (as[iGlobal] >> bit) & 1;
    }
}

__kernel void radix_gather(
    __global unsigned int *sums, 
    const unsigned int n,
    const unsigned int globalStep
) {
    const size_t iGlobal = (get_global_id(0) + 1) * globalStep;
    const size_t iGroup = get_local_id(0);
    
    __local unsigned int sumsLocal[LOCAL_SIZE];
   
    if (iGlobal <= n) {
        sumsLocal[iGroup] = sums[iGlobal];
    } else {
        sumsLocal[iGroup] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (size_t step = 1; step <= LOCAL_SIZE; step <<= 1) {
        unsigned int valueToAdd = 0;
        if (step <= iGroup && iGlobal <= n) {
            valueToAdd = sumsLocal[iGroup - step];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

#ifdef DEBUG_OUTPUT
        if (step <= iGroup && iGlobal <= n)
            printf("\tgathering %d from %d to %d [%d total]", valueToAdd, iGlobal - step * globalStep, iGlobal, sumsLocal[iGroup] + valueToAdd);
#endif
        
        sumsLocal[iGroup] += valueToAdd;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (iGlobal <= n) {
        sums[iGlobal] = sumsLocal[iGroup];
    }
}

__kernel void radix_propagate(
    __global unsigned int *sums, 
    const unsigned int n,
    const unsigned int globalStep,
    __global unsigned int *sums_next
) {
    const size_t iGlobal = (get_global_id(0) + 1) * globalStep;
    const size_t prevGroupEnd = get_global_offset(0) * globalStep;
    
    const unsigned int prefixSum = sums[prevGroupEnd];

    if (get_global_id(0) == 0) {
        sums_next[0] = 0;
    }

    if (iGlobal <= n) {
        sums_next[iGlobal] = sums[iGlobal] + prefixSum;
        
#ifdef DEBUG_OUTPUT
        printf("\tpropagating %d from %d to %d [%d total]", prefixSum, prevGroupEnd, iGlobal, sums[iGlobal] + prefixSum);
#endif
    }
}

__kernel void radix_move(
    const __global unsigned int *as, 
    const unsigned int n, 
    const __global unsigned int *a_cnts, 
    __global unsigned int *bs
) {
    const size_t iGlobal = get_global_id(0);

    const size_t zeros = n - a_cnts[n];

#ifdef DEBUG_OUTPUT
    if (iGlobal == 0) {
        printf("%d zeros", zeros);
    }
#endif

    if (iGlobal < n) {
        const unsigned int ones_on_prefix = a_cnts[iGlobal];
        const size_t index = 
            a_cnts[iGlobal + 1] == ones_on_prefix // 'is zero'
            ? iGlobal - ones_on_prefix
            : zeros + ones_on_prefix;

#ifdef DEBUG_OUTPUT
        printf("%d goes from %d to %d", as[iGlobal], iGlobal, index);
#endif

        bs[index] = as[iGlobal];
    }

    // const size_t ones = a_cnts[n];
    // const size_t zeros = n - ones;

    // const size_t isZero = iGlobal + ones < n;
    // const size_t indexToSearch = isZero ? iGlobal - a_cnts[iGlobal] : zeros + a_cnts[iGlobal];
    
    // size_t l = 0;
    // size_t r = n + 1;
    // while (l + 1 < r) {
    //     const size_t m = (l + r) >> 1;
        
    //     size_t index;
    //     if (isZero) {
    //         index = m - a_cnts[m];
    //     } else {
    //         index = zeros + a_cnts[m];
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
