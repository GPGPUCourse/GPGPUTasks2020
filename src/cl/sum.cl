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

    __local unsigned int localSum[WORK_GROUP_SIZE];
    localSum[localIndex] = 0;
    for (int i = 0; i < RANGE_PER_WORK_ITEM; ++i) {
        unsigned int id = group_id * WORK_GROUP_SIZE * RANGE_PER_WORK_ITEM 
                                + i * WORK_GROUP_SIZE + localIndex;
        if (id < n)
            localSum[localIndex] += a[id];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (unsigned int curShift = WORK_GROUP_SIZE / 2; curShift > 0; curShift >>= 1){
        if (localIndex < curShift)
            localSum[localIndex] += localSum[localIndex + curShift];
        if (localIndex > WARP_SIZE)
            barrier(CLK_LOCAL_MEM_FENCE);    
    }
    if (localIndex == 0)
        res[group_id] = localSum[0];
}