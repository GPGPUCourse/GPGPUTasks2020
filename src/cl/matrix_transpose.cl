#define TILE_SIZE 16

__kernel void matrix_transpose(__global const float *a, __global float *at, unsigned int m, unsigned int k) {
    unsigned int x = get_global_id(0);
    unsigned int y = get_global_id(1);
    __local float tile[TILE_SIZE][TILE_SIZE + 1];
    unsigned int local_x = get_local_id(0);
    unsigned int local_y = get_local_id(1);
    tile[local_y][local_x] = (x < k && y < m) ? a[y * k + x] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    unsigned int xn = x + local_y - local_x;
    unsigned int yn = y + local_x - local_y;
    if (xn < k && yn < m) at[xn * m + yn] = tile[local_x][local_y];
}