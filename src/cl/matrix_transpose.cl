__kernel void matrix_transpose(__global const float* as, __global float* as_t,
                               unsigned int K, unsigned int M)
{
    __local float tile[TILE_SIZE][TILE_SIZE + FIX_BANK];
    const size_t item_i = get_group_id(0);
    const size_t item_j = get_group_id(1);

    const size_t local_i = get_local_id(0);
    const size_t local_j = get_local_id(1);

    // tile coords are (item_i, item_j)
    const size_t from_i = item_i * TILE_SIZE + local_i;
    const size_t from_j = item_j * TILE_SIZE + local_j;

    // read access should be coalesced
    // item_i/item_j are fixed in group
    // from_i grows with local_id(0)
    if (from_i < K && from_j < M) {
        tile[local_j][local_i] = as[from_j * K + from_i];
    }

    // wait whole group
    barrier(CLK_LOCAL_MEM_FENCE);

    // transpose tile coords to (item_j, item_i)
    const size_t to_i = item_j * TILE_SIZE + local_i;
    const size_t to_j = item_i * TILE_SIZE + local_j;

    // write access should be coalesced
    // item_i/item_j are fixed in group
    // to_i grows with local_id(0)
    if (to_i < M && to_j < K) {
        // transpose happens when we swap local_i and local_j
        as_t[to_j * M + to_i] = tile[local_i][local_j];
    }
}