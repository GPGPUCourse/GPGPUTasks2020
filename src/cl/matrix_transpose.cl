#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void matrix_transpose(__global const float *A, __global float *A_T, unsigned int M, unsigned int K)
{
    int id0 = get_global_id(0);
    int id1 = get_global_id(1);
    int lid0 = get_local_id(0);
    int lid1 = get_local_id(1);
    int g0 = get_group_id(0);
    int g1 = get_group_id(1);
    
    __local float lA[WORK_GROUP_SIDE][WORK_GROUP_SIDE + 1];
    
    lA[lid0][lid1] = A[id1 * K + id0];
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int offs = g0 * WORK_GROUP_SIDE * M + g1 * WORK_GROUP_SIDE;
    A_T[offs + lid1 * M + lid0] = lA[lid1][lid0];
}