__kernel void matrix_multiplication(
    __global const float *a, __global const float *b_transposed, __global float *c, 
    const unsigned Y, const unsigned Z, const unsigned X
) {
    // WG x WG groups
    const size_t sideOfGroup = get_local_size(1); // same as local_size(0)

    const size_t yInGroup = get_local_id(0);
    const size_t xInGroup = get_local_id(1);

    // correspond to WARP x WARP == (k * WG) x (k * WG) tiles
    const size_t yGroup = get_group_id(0);
    const size_t xGroup = get_group_id(1);

    // mapping from group index to position from tile start
    const size_t indexInGroup = yInGroup * sideOfGroup + xInGroup;
    const size_t xShiftInTile = indexInGroup % TILE_SIDE; 
    const size_t yShiftInTile = indexInGroup / TILE_SIDE; 

    // iterations
    const size_t zTiles = (Z + TILE_SIDE - 1) / TILE_SIDE;
    const size_t warpCount = sideOfGroup * sideOfGroup / TILE_SIDE;

    __local float a_tile[TILE_SIDE][TILE_SIDE + 1]; // +1 for memory bank shifting
    __local float b_tile[TILE_SIDE][TILE_SIDE + 1];
    __local float c_tile[TILE_SIDE][TILE_SIDE + 1];

    // 0. initialize result
    for (size_t rowsProcessed = 0; rowsProcessed < TILE_SIDE; rowsProcessed += warpCount) {
        const size_t yInTile = rowsProcessed + yShiftInTile;
        const size_t xInTile =                 xShiftInTile;

        c_tile[yInTile][xInTile] = 0;
    }

    for (size_t zTile = 0; zTile < zTiles; ++zTile) {
        // 1. load tiles to local memory
        for (size_t rowsProcessed = 0; rowsProcessed < TILE_SIDE; rowsProcessed += warpCount) {
            const size_t yInTile = rowsProcessed + yShiftInTile;
            const size_t xInTile =                 xShiftInTile;

            const size_t yGlobal = yGroup * TILE_SIDE + yInTile;
            const size_t xGlobal = zTile  * TILE_SIDE + xInTile;
            
            if (yGlobal < Y && xGlobal < Z)
                a_tile[yInTile][xInTile] = a[yGlobal * Z + xGlobal];
            else
                a_tile[yInTile][xInTile] = 0;
        }

        for (size_t rowsProcessed = 0; rowsProcessed < TILE_SIDE; rowsProcessed += warpCount) {
            const size_t yInTile = rowsProcessed + yShiftInTile;
            const size_t xInTile =                 xShiftInTile;

            const size_t yGlobal = xGroup * TILE_SIDE + yInTile;
            const size_t xGlobal = zTile  * TILE_SIDE + xInTile;
        
            if (yGlobal < X && xGlobal < Z)
                b_tile[yInTile][xInTile] = b_transposed[yGlobal * Z + xGlobal];
            else
                b_tile[yInTile][xInTile] = 0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // 2. multiply them
        for (size_t rowsProcessed = 0; rowsProcessed < TILE_SIDE; rowsProcessed += warpCount) {
            const size_t yInTile = rowsProcessed + yShiftInTile;
            const size_t xInTile =                 xShiftInTile;

            for (size_t zInTile = 0; zInTile < TILE_SIDE; ++zInTile)
                c_tile[yInTile][xInTile] += a_tile[yInTile][zInTile] * b_tile[xInTile][zInTile];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // 3. write result to global memory
    for (size_t rowsProcessed = 0; rowsProcessed < TILE_SIDE; rowsProcessed += warpCount) {
        const size_t yInTile = rowsProcessed + yShiftInTile;
        const size_t xInTile =                 xShiftInTile;

        const size_t yGlobal = yGroup * TILE_SIDE + yInTile;
        const size_t xGlobal = xGroup * TILE_SIDE + xInTile;
        
        if (yGlobal < Y && xGlobal < X)
            c[yGlobal * X + xGlobal] = c_tile[yInTile][xInTile];
    }
}