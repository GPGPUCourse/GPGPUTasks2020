#ifdef USE_LOCAL
#   define from temp
#   define to temp
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
    __local float temp[LOCAL_SIZE];

    temp[iLocal] = as[iGlobal];
    barrier(CLK_LOCAL_MEM_FENCE);

    step = 1;

while (step < LOCAL_SIZE) {
#endif

    const unsigned int offset = i & (~((step << 1) - 1));
    
    // TODO

#ifdef USE_LOCAL
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
    to[i] = foundElement;

#ifdef USE_LOCAL
    barrier(CLK_LOCAL_MEM_FENCE);
    
    step <<= 1;
} // while

    as_next[iGlobal] = temp[iLocal];
#endif
}
