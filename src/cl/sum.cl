#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 256

__kernel void sum(__global unsigned int* as,
                  __global unsigned int* bs,
                  unsigned int n)
{
    unsigned int i = get_global_id(0);
    unsigned int loci = get_local_id(0);

    __local unsigned int buf[WORK_GROUP_SIZE];

    buf[loci] = (i < n ? as[i] : 0);

    barrier(CLK_LOCAL_MEM_FENCE);
    if (loci == 0) {
        unsigned int s = 0;
        for (int j = 0; j < WORK_GROUP_SIZE; ++j) {
            s += buf[j];
        }
        bs[i / WORK_GROUP_SIZE] = s;
    }
}
