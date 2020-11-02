#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void radix(__global unsigned int* from,
                    __global unsigned int* to,
                    __global unsigned int* prefixSums,
                    unsigned int zerosCount,
                    unsigned int shift,
                    unsigned int n)
{
    unsigned int id = get_global_id(0);
    if (id < n) {
        unsigned int x = from[id];
        unsigned int zerosBefore = prefixSums[id];
        unsigned int i = zerosBefore;
        if ((x >> shift) & 1)
            i = zerosCount + (id - zerosBefore);
        else
            i--;
        to[i] = x;
    }
}
