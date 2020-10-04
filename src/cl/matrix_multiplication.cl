__kernel void matrix_multiplication(__global const float* as,
                                    __global const float* bs,
                                    __global float* cs,
                                    unsigned int M, unsigned int K, unsigned int N)
{
    __local float tile[TILE_SIZE][TILE_SIZE];
    __local float a[TILE_SIZE][TILE_SIZE];
    // +1 to fix bank conflict when we read b by column
    __local float b[TILE_SIZE][TILE_SIZE + 1];
    const size_t tile_i = get_group_id(0);
    const size_t tile_j = get_group_id(1);

    const size_t local_i = get_local_id(0);
    const size_t local_j = get_local_id(1);
    tile[local_j][local_i] = 0;

    // +1 if tile goes beyond border
    const size_t tiles = (K / TILE_SIZE) + 1;

    for (size_t tile_idx = 0; tile_idx < tiles; ++tile_idx) {
        // we take tile from as with coords (tile_idx, tile_j)
        const size_t as_i = tile_idx * TILE_SIZE + local_i;
        const size_t as_j = tile_j * TILE_SIZE + local_j;

        if (as_i < K && as_j < M) {
            a[local_j][local_i] = as[as_j * K + as_i];
        } else {
            a[local_j][local_i] = 0;
        }

        // we take tile from bs with coords (tile_i, tile_idx)
        const size_t bs_i = tile_i * TILE_SIZE + local_i;
        const size_t bs_j = tile_idx * TILE_SIZE + local_j;

        if (bs_i < N && bs_j < K) {
            b[local_j][local_i] = bs[bs_j * N + bs_i];
        } else {
            b[local_j][local_i] = 0;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // we should take (x, local_j) from a
        // and (local_i, y) from b
        for (size_t idx = 0; idx < TILE_SIZE; ++idx) {
            tile[local_j][local_i] += a[local_j][idx] * b[idx][local_i];
        }
    }

    // cs tile coords are (tile_i, tile_j)
    const size_t cs_i = tile_i * TILE_SIZE + local_i;
    const size_t cs_j = tile_j * TILE_SIZE + local_j;
    if (cs_i < N && cs_j < M) {
        cs[cs_j * N + cs_i] = tile[local_j][local_i];
    }
}