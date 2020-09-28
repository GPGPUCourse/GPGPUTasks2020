#ifdef __CLION_IDE__


#include "clion_defines.cl"
#endif

#line 8

__kernel void aplusb(__global const float* as,
                     __global const float* bs,
                     __global float* cs,
                     unsigned int n)
{
    const size_t idx = get_global_id(0);

    if (idx >= n) {
        return;
    }

    cs[idx] = as[idx] + bs[idx];
}
