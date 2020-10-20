#define TILE_SIZE 16

__kernel void matrix_transpose(__global const float* a, __global float* at, int n, int m) {
    const int global_i = get_global_id(0);
    const int global_j = get_global_id(1);
    const int local_i = get_local_id(0);
    const int local_j = get_local_id(1);
	__local float tile[TILE_SIZE][TILE_SIZE + 1];

	if (global_i < m && global_j < n) {
        tile[local_i][local_j] = a[global_i + global_j * m];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int tile_i = get_group_id(1) * TILE_SIZE + local_i;
    int tile_j = get_group_id(0) * TILE_SIZE + local_j;
    if (tile_i < n && tile_j < m) {
        at[tile_j * n + tile_i] = tile[local_j][local_i];
    }
}

