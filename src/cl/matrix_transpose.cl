#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define TILE_SIZE 16
__kernel void naive_matrix_transpose(__global const float* a,
                                     __global       float* at,
                                     unsigned int m, 
                                     unsigned int k)
{
    int global_i = get_global_id(0);
    int global_j = get_global_id(1);

    if (global_i < k && global_j < m)
    {
        float x = a[global_j * k + global_i];
        at[global_i * m + global_j] = x;
    }
}

__kernel void matrix_transpose(__global const float* a,
                               __global       float* at,
                               unsigned int m, 
                               unsigned int k)
{
    int global_i = get_global_id(0);
    int global_j = get_global_id(1);

    __local float tile[TILE_SIZE * TILE_SIZE];
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    if (global_i < k && global_j < m)
        tile[local_j * TILE_SIZE + local_i] = a[global_j * k + global_i];

    barrier(CLK_LOCAL_MEM_FENCE); 

    int group_i = get_group_id(0);
    int group_j = get_group_id(1);

    if (global_i < k && global_j < m)
        at[(group_i * TILE_SIZE + local_j) * m  + group_j * TILE_SIZE + local_i] = tile[local_i * TILE_SIZE + local_j];
}
