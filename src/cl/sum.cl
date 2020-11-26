#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 128
__kernel void sum(__global const unsigned int* as, __global unsigned int* res, const unsigned int n) {

    const unsigned int id = get_global_id(0);
    const unsigned int l_id = get_local_id(0);

    __local unsigned int l_as[WORK_GROUP_SIZE];

    if (n > id)
        l_as[l_id] = as[id];
    else
        l_as[l_id] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = WORK_GROUP_SIZE; i > 1; i /= 2) {
        if (2 * l_id < i) {
            unsigned int a = l_as[l_id];
            unsigned int b = l_as[l_id + i / 2];
            l_as[l_id] = a + b;
        }

        barrier(CLK_LOCAL_MEM_FENCE); // wait for tree summation update
    }

    if (l_id == 0)
        atomic_add(res, l_as[0]);
}
