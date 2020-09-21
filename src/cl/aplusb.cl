#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

#line 8

__kernel void aplusb(__global const float* a, __global const float* b, __global float* c, unsigned int n)
{
    unsigned int i = get_global_id(0);
    if (i >= n)
        return;
    c[i] = a[i] + b[i];
}
