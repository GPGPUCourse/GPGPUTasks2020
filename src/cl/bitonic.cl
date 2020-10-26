__kernel void bitonic(__global float *as, const unsigned int n, const unsigned int globalStep, const unsigned int globalSubStep)
{
    const size_t iGlobal = get_global_id(0);

#ifdef LOCAL_SIZE                                                              // we want to sort locally as much as possible
    const size_t iGroup = get_local_id(0);
    const size_t iOfGroup = get_global_offset(0);
    const size_t nLocal = min(n - iOfGroup, (size_t) LOCAL_SIZE);

    __local float asLocal[LOCAL_SIZE];

    asLocal[iGroup] = as[iGlobal];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (size_t step = 1; step <= globalStep; ++step) {                        //    sorting by boxes with changing order
        bool isRight = (iGroup >> step) & 1;                                   //    determine box type

        for (int subStep = step - 1; subStep >= 0; --subStep) {

            const size_t delta = (iGroup >> subStep) & 1;
            const size_t iGroupPair = iGroup + ((1 - 2 * delta) << subStep);   //    find another end of the arrow

            const float x     = asLocal[iGroup    ];
            const float xPair = asLocal[iGroupPair];
            if ((1 << subStep) >= WARP_SIZE) {                                 //    no sync required for smaller boxes
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            
            if (x != xPair && (delta ^ isRight) == (x < xPair)) {              //    if order is incorrect -- swap
                asLocal[iGroup] = xPair;
            }
            if ((1 << subStep) >= WARP_SIZE) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);                                          //    sync between local iterations
    }

    as[iGlobal] = asLocal[iGroup];
#else                                                                          // steps now are big enough, sorting in global memory (same algorithm, but single step)
    const bool   isRight = (iGlobal >> (globalStep - 1)) & 1;
    
    const size_t iMapped = iGlobal + ((iGlobal >> (globalSubStep - 1)) << (globalSubStep - 1));
    const size_t iMappedPair = iMapped + (1 << (globalSubStep - 1));

    if (iMappedPair < n) {
        const float x     = as[iMapped    ];
        const float xPair = as[iMappedPair];
        
        if (x != xPair && isRight == (x < xPair)) {
            as[iMapped    ] = xPair;
            as[iMappedPair] = x;
        }
    }
#endif
}
