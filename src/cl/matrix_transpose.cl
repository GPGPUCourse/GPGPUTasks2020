#define TILE_SIZE 32

__kernel void matrix_transpose(__global const float* a, __global float* at, int n, int m) {
    const int global_i = get_global_id(0);
    const int global_j = get_global_id(1);
    const int local_i = get_local_id(0);
    const int local_j = get_local_id(1);
	__local float tile[TILE_SIZE][TILE_SIZE + 1];

	if (global_i < n && global_j < m) {
        tile[local_j][local_i] = a[global_i + global_j * m];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (global_i < n && global_j < m) {
        int tile_i = get_group_id(0) * TILE_SIZE + local_j;
        int tile_j = get_group_id(1) * TILE_SIZE + local_i;
        at[tile_i * m + tile_j] = tile[local_i][local_j];
    }
}

