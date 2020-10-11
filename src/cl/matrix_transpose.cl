#define TILE_SIDE WARP_SIZE

__kernel void matrix_transpose(__global const float *matrix, __global float *matrix_transposed, const unsigned Y, const unsigned X)
{
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

    __local float tile[TILE_SIDE][TILE_SIDE + 1]; // +1 for memory bank shifting

    // 1. load to local memory
    const size_t warpCount = sideOfGroup * sideOfGroup / TILE_SIDE;
    
    for (size_t rowsProcessed = 0; rowsProcessed < TILE_SIDE; rowsProcessed += warpCount) {
        const size_t yInTile = rowsProcessed + yShiftInTile;
        const size_t xInTile =                 xShiftInTile;

        const size_t yGlobal = yGroup * TILE_SIDE + yInTile;
        const size_t xGlobal = xGroup * TILE_SIDE + xInTile;
        
        if (yGlobal < Y && xGlobal < X)
            tile[yInTile][xInTile] = matrix[yGlobal * X + xGlobal];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 2. transpose tiles block by block
    // [ A, B ] ^ T - [ A ^ T, C ^ T ]
    // [ C, D ]     - [ B ^ T, D ^ T ]

    // 2.1. start with non-diagonal subtiles
    for (size_t yOfSubtile = 0; yOfSubtile < TILE_SIDE; yOfSubtile += sideOfGroup) {
        const size_t yInTile  = yOfSubtile + yInGroup;
        
        for (size_t xOfSubtile = yOfSubtile + sideOfGroup; xOfSubtile < TILE_SIDE; xOfSubtile += sideOfGroup) {
            const size_t xInTile  = xOfSubtile + xInGroup;
        
            const float element  = tile[yInTile][xInTile];
            const float elementT = tile[xInTile][yInTile];
            
            tile[yInTile][xInTile] = elementT;
            tile[xInTile][yInTile] = element ;
            
            // no barriers needed because of tile independence
        }
    }

    // 2.2. finish with diagonal ones
    for (size_t xyOfSubtile = 0; xyOfSubtile < TILE_SIDE; xyOfSubtile += sideOfGroup) {
        const float element = tile[xyOfSubtile + yInGroup][xyOfSubtile + xInGroup];
        barrier(CLK_LOCAL_MEM_FENCE);
                          
        tile[xyOfSubtile + xInGroup][xyOfSubtile + yInGroup] = element;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // 3. write back to global memory
    for (size_t rowsProcessed = 0; rowsProcessed < TILE_SIDE; rowsProcessed += warpCount) {
        const size_t yInTile = rowsProcessed + yShiftInTile;
        const size_t xInTile =                 xShiftInTile;

        const size_t yGlobal = xGroup * TILE_SIDE + yInTile;
        const size_t xGlobal = yGroup * TILE_SIDE + xInTile;
        
        if (yGlobal < X && xGlobal < Y)
            matrix_transposed[yGlobal * Y + xGlobal] = tile[yInTile][xInTile];
    }
}