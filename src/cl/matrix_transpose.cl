#define TILE_SIZE 16

__kernel void matrix_transpose(__global const float* as,
                               __global       float* as_t,
                               unsigned int M,
                               unsigned int K)
{
    __local float tile[TILE_SIZE][TILE_SIZE+1]; // +1 shift to resolve bank conflict

    int glob_i = get_global_id(0);
    int glob_j = get_global_id(1);
    int i = get_local_id(0);
    int j = get_local_id(1);

    // read matrix to local memory
    if (glob_i < M && glob_j < K)
    {
        tile[j][i] = as[glob_i + glob_j * K];
    }
    // wait for all threads
    barrier(CLK_LOCAL_MEM_FENCE);

    // write back to vram
    if (glob_i < M && glob_j < K)
    {
        as_t[(get_group_id(0) * TILE_SIZE + j) * K + (get_group_id(1) * TILE_SIZE + i)] = tile[i][j];
    }
} 