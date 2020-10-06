#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void sum(
    __global unsigned int* from,
    __global unsigned int* to,
    unsigned int n,
    unsigned int valsInStep)
{
    size_t localId = get_local_id(0);
    size_t globalId = get_global_id(0);

    int toIdx = globalId;
    int fromIdxStart = toIdx*valsInStep;
    int fromIdxEnd = min((toIdx+1)*valsInStep, n);
    unsigned int sum = 0;
    for (int i = fromIdxStart; i < fromIdxEnd; ++i) {
        sum += from[i];
    }
    to[toIdx] = sum;
}
