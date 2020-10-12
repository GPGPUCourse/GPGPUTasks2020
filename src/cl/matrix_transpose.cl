#define TILE_SIZE 16

__kernel void matrix_transpose(__global float *a, __global float *at, unsigned int m, unsigned int k)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int l_i = get_local_id(0);
    int l_j = get_local_id(1);
    int gr_i = get_group_id(0);
    int gr_j = get_group_id(1);

    __local float tile[TILE_SIZE][TILE_SIZE + 1];

    if (i < m && j < k) {
        tile[l_j][l_i] = a[j * k + i];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (i < m && j < k) {
        at[(gr_i * TILE_SIZE + l_j) * k + (gr_j * TILE_SIZE + l_i)] = tile[l_i][l_j];
    }
}