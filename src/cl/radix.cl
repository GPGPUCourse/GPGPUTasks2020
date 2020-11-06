#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define P 2
#define MOD (1 << P)
#define WORK_GROUP_SIZE 256

__kernel void radix(__global unsigned int* as, unsigned int pos) {
    unsigned int local_id = get_local_id(0);
    unsigned int global_id = get_local_id(0);

    __local unsigned int local_as[WORK_GROUP_SIZE][MOD - 1];
    for (int i = 0; i < MOD - 1; ++i) {
        unsigned int cut = as[global_id] >> (pos * P);
        local_as[local_id][i] = (cut % MOD == i) ? 1 : 0;
    }

    
}
