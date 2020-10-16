#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6
#define TILE_SIZE 16

__kernel void matrix_transpose(__global float* a, __global float* at, unsigned int m, unsigned int k) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    int local_x = get_local_id(0);
    int local_y = get_local_id(1);

    int group_x = get_group_id(0);
    int group_y = get_group_id(1);

    __local float tile[TILE_SIZE][TILE_SIZE + 1];

    if (x < k && y < m) {
        tile[local_y][local_x] = a[y * k + x];
    }
    barrier(CLK_LOCAL_MEM_FENCE);


    int new_arr_x = group_y * get_local_size(0) + local_x;
    int new_arr_y = group_x * get_local_size(1) + local_y;

    if (new_arr_x < m && new_arr_y < k) {
        at[new_arr_y * m + new_arr_x] = tile[local_x][local_y];
    }
}