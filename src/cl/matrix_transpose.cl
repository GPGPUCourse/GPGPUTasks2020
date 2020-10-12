#define TILE_SIZE 16

__kernel void matrix_transpose(__global float* a,
                               __global float* at,
                               unsigned int rows,
                               unsigned int cols)
{
    unsigned int gx = get_global_id(0);
    unsigned int gy = get_global_id(1);

    __local float localA[TILE_SIZE][TILE_SIZE];

    unsigned int lx = get_local_id(0);
    unsigned int ly = get_local_id(1);

    if(gx + gy * cols < rows*cols) {
        localA[lx][ly] = a[gx + gy * cols];
    }

    __local unsigned int blockFirstIndex;
    blockFirstIndex = (get_group_id(0) * rows + get_group_id(1)) * TILE_SIZE;

    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int itemIndex = blockFirstIndex + lx + ly * rows;

    at[blockFirstIndex + lx + ly * rows] = localA[ly][lx];
    if (itemIndex < rows*cols) {
        at[itemIndex] = localA[ly][lx];
    }
}

//__kernel void matrix_transpose(__global float* a,
//                               __global float* at,
//                               unsigned int rows,
//                               unsigned int cols)
//{
//    unsigned int gx = get_global_id(0);
//    unsigned int gy = get_global_id(1);
//
//    __local float localA[TILE_SIZE][TILE_SIZE];
//
//    unsigned int lx = get_local_id(0);
//    unsigned int ly = get_local_id(1);
//
//    if(gx + gy * cols < rows*cols) {
//        localA[lx][ly] = a[gx + gy * cols];
//    }
//
//    __local unsigned int blockFirstIndex;
//    blockFirstIndex = (get_group_id(0) * rows + get_group_id(1)) * TILE_SIZE;
//
//    barrier(CLK_LOCAL_MEM_FENCE);
//
//    unsigned int itemIndex = blockFirstIndex + lx + ly * rows;
//
//    at[blockFirstIndex + lx + ly * rows] = localA[ly][lx];
//    if (itemIndex < rows*cols) {
//        at[itemIndex] = localA[ly][lx];
//    }
//}