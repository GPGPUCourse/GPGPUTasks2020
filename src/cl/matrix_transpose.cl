#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

#define TILE_SIZE 16
__kernel void matrix_transpose(__global const float *a,
                               __global float *a_t,
                               unsigned int M,
                               unsigned int K) {
    int i = get_global_id(0); //  column
    int j = get_global_id(1); //  row
    int local_x = get_local_id(0);
    int local_y = get_local_id(1);

    __local float tile[TILE_SIZE][TILE_SIZE + 1];
    barrier(CLK_LOCAL_MEM_FENCE);
    tile[local_x][local_y] = a[j * K + i];

    if (i < K && j <  M)
        a_t[i * M + j] = tile[local_x][local_y];
}