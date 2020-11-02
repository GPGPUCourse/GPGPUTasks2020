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
    const size_t iGlobal = (get_global_id(0) + 1) * globalStep - 1;
    const size_t iGroup = get_local_id(0);
    
    __local unsigned int sumsLocal[LOCAL_SIZE];
   
    if (iGlobal < n) {
        sumsLocal[iGroup] = sums[iGlobal];
    } else {
        sumsLocal[iGroup] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (size_t step = 1; step < LOCAL_SIZE; step <<= 1) {
        unsigned int valueToAdd = 0;
        if (step <= iGroup) {
            valueToAdd = sumsLocal[iGroup - step];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

#if LOG_LEVEL > 2
        if (step <= iGroup && iGlobal < n)
            printf("\tgathering %d from %d to %d [%d total]", valueToAdd, iGlobal - step * globalStep, iGlobal, sumsLocal[iGroup] + valueToAdd);
#endif
        
        sumsLocal[iGroup] += valueToAdd;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

#if LOG_LEVEL == 2
    if (iGlobal < n)
        printf("\tgathered %d at %d[%d]", sumsLocal[iGroup], iGlobal, iGroup);
#endif

    if (iGlobal < n) {
        sums[iGlobal] = sumsLocal[iGroup];
    }
}

__kernel void radix_propagate(
    const __global unsigned int *sums, 
    const unsigned int n,
    const unsigned int globalStep,
    __global unsigned int *sums_next
) {
    const size_t iGlobal = (get_global_id(0) + 1) * globalStep - 1;
    const size_t iGroup = get_local_id(0);

    const size_t macroStep = globalStep * LOCAL_SIZE;
    const size_t macroStepIndex = iGlobal % macroStep;
    const int prevPrefix = macroStepIndex + 1 != macroStep ? iGlobal - macroStepIndex - 1 : -1;
    
    if (iGlobal < n) {    
        const unsigned int prevSum = prevPrefix >= 0 ? sums[prevPrefix] : 0;

#if LOG_LEVEL == 2
        if (stepIndex + 1 == globalStep) {
            printf("\tpropagating %d from %d[%d]", sums[iGlobal], iGlobal, get_local_id(0));
        }
#endif

        sums_next[iGlobal] = sums[iGlobal] + prevSum;
        
#if LOG_LEVEL > 2
        printf("\tpropagating %d from %d to %d [%d total]", prevSum, prevPrefix, iGlobal, sums[iGlobal] + prevSum);
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

#if LOG_LEVEL > 1
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

#if LOG_LEVEL > 2
        printf("%d goes from %d to %d", as[iGlobal], iGlobal, index);
#endif
        if (index < n) {
            bs[index] = as[iGlobal];
        } else {
#if LOG_LEVEL > 2
            printf("%d tried to go from %d to invalid %d", as[iGlobal], iGlobal, index);
#endif
        }
    }
}
