#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void sum_naive(__global const unsigned int *as, __global unsigned int *res, unsigned int n)
{
    const int work_group_size = get_local_size(0);
    __local int A[WORK_GROUP_SIZE];
    
    int id = get_global_id(0);
    int local_id = get_local_id(0);
    
    A[local_id] = as[id];
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (local_id == 0) {
        unsigned int s = 0;
        for (int i = 0; i < work_group_size; ++i) {
            s += A[i];
        }
    
        atomic_add(res, s);
    }
}


__kernel void sum_tree(__global const unsigned int *as, __global unsigned int *res, unsigned int n)
{
    const int work_group_size = get_local_size(0);
    __local int A[WORK_GROUP_SIZE];
    __local int B[WORK_GROUP_SIZE];
    
    int id = get_global_id(0);
    int local_id = get_local_id(0);
    
    A[local_id] = as[id];
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (int gap = WORK_GROUP_SIZE; gap > 1; gap /= 2) {
        if (2 * local_id < gap) {
            int a = A[local_id];
            int b = A[local_id + gap / 2];
            A[local_id] = a + b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (local_id == 0) {
        atomic_add(res, A[0]);
    }
}