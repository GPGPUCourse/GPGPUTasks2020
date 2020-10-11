#define TILE_SIZE 16

__kernel void matrix_transpose(__global const float* a, __global float* a_t, unsigned int m, unsigned int k)
{
    const size_t i = get_global_id(0);
    const size_t j = get_global_id(1);

    const size_t local_i = get_local_id(0);
    const size_t local_j = get_local_id(1);

    const size_t group_i = get_group_id(0);
    const size_t group_j = get_group_id(1);

    // add "fake" elements as the last column
    __local float tile[TILE_SIZE][TILE_SIZE + 1];

    if (i < m && j < k)
        tile[local_j][local_i] = a[i + j * k];
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (i < m && j < k)
        a_t[(group_i * TILE_SIZE + local_j) * k + (group_j * TILE_SIZE + local_i)] = tile[local_i][local_j];
}