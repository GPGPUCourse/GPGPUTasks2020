#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define LOG_GROUP_SIZE (7)
#define GROUP_SIZE (1 << LOG_GROUP_SIZE)

__kernel void prefix(__global unsigned int* from,
                    __global unsigned int* to,
                    unsigned int stage,
                    unsigned int shift,
                    unsigned int n)
{
    unsigned int id = get_global_id(0);
    if (stage == 0) {
        if (id < n)
            to[id] = ((from[id] >> shift) & 1) ? 0 : 1;
    } else if (stage == 1){
        // __local unsigned cache[GROUP_SIZE];
        // unsigned int localID = get_local_id(0);
        // if (id < n)
        //     cache[localID] = from[id];
        // else
        //     cache[localID] = 0;
        // barrier(CLK_LOCAL_MEM_FENCE);
        // unsigned int current = GROUP_SIZE >> 1;
        // while (current > 0) {
        //     if (localID < current)
        //         cache[localID] += cache[localID + current];
        //     current >>= 1;
        //     barrier(CLK_LOCAL_MEM_FENCE);
        // }
        // if (localID == 0 && id < n){
        //     to[id >> LOG_GROUP_SIZE] = cache[localID];
        // }
    } else {
        __local unsigned cache[GROUP_SIZE];
        unsigned int localID = get_local_id(0);
        if (id < n)
            cache[localID] = from[id];
        else
            cache[localID] = 0;
        barrier(CLK_LOCAL_MEM_FENCE);
        unsigned int current = 0;
        while ((1 << current) < GROUP_SIZE) {
            if ((localID >> current) & 1)
                cache[localID] += cache[((localID >> current) << current) - 1];
            current++;
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        unsigned int res = cache[localID];
        if ((id >> LOG_GROUP_SIZE) > 0 && id < n)
            res += to[(id >> LOG_GROUP_SIZE) - 1];
        if (id < n)
            from[id] = res;
    }
}
