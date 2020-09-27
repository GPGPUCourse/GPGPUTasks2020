#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE (128)
#define RANGE_PER_WORK_ITEM (64)

__kernel void summa(__global const unsigned int* a,
                     __global unsigned int* res,
                     unsigned int n)
{
    unsigned int group_id = get_group_id(0);
    unsigned int localIndex = get_local_id(0);

    unsigned int index = get_global_id(0);
    __local unsigned int localSum[WORK_GROUP_SIZE];
    for (int i = 0; i < RANGE_PER_WORK_ITEM; ++i) {
        unsigned int id = group_id * WORK_GROUP_SIZE * RANGE_PER_WORK_ITEM 
                                + i * WORK_GROUP_SIZE + localIndex;
        if (id < n)
            localSum[localIndex] += a[id];
    }
    unsigned int curShift = WORK_GROUP_SIZE / 2;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (localIndex < 64)
        localSum[localIndex] += localSum[localIndex + 64];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (localIndex < 32)
        localSum[localIndex] += localSum[localIndex + 32];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (localIndex < 16)
        localSum[localIndex] += localSum[localIndex + 16];
    barrier(CLK_LOCAL_MEM_FENCE);
    int t = 16;

    if (localIndex < 8){
        localSum[localIndex] = localSum[localIndex + 8];
        t = 8;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (localIndex == 0){
        for (unsigned int i = 1; i < t; i++)
            localSum[0] += localSum[i];
        atomic_add(res, localSum[0]);
    }
}