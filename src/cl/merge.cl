#ifdef USE_LOCAL
#   define from temp[current]
#   define to temp[current ^ 1]
#   define i iLocal
#else
#   define from as
#   define to as_next
#   define i iGlobal
#endif

__kernel void merge(const __global float *as, __global float *as_next, const unsigned int n, unsigned int step) {
    const unsigned int iGlobal = get_global_id(0);
    const unsigned int iLocal = get_local_id(0);
    
#ifdef USE_LOCAL
    __local float temp[2][LOCAL_SIZE];

    temp[0][iLocal] = as[iGlobal];
    barrier(CLK_LOCAL_MEM_FENCE);

    step = 1;
    unsigned int current = 0;

while (step < LOCAL_SIZE) {
#endif

    const unsigned int offset[2] = {
         i & (~((step << 1) - 1)),
        (i & (~((step << 1) - 1))) + step
    };
    
    bool isFirstHalf = i < offset[1];
    const float iValue = from[i];

    int l = offset[isFirstHalf] - 1;
    int r = offset[isFirstHalf] + step;

    while (l + 1 < r) {
        const int m = (l + r) >> 1;
        const float mValue = from[m];

        if (mValue < iValue || (!isFirstHalf && mValue == iValue)) {
            l = m;
        } else {
            r = m;
        }
    }

    const int iNext = offset[0] + (i - offset[!isFirstHalf]) + (r - offset[isFirstHalf]);
    to[iNext] = iValue;

#ifdef USE_LOCAL
    barrier(CLK_LOCAL_MEM_FENCE);
    
    current ^= 1;
    step <<= 1;
} // while

    as_next[iGlobal] = temp[current][iLocal];
#endif
}
