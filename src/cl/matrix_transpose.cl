 
#define TILE_SIZE 16

__kernel void matrix_transpose(__global const float *A, __global float * A_t,
                                unsigned int M, unsigned int K)
{
    const unsigned int gx = get_global_id(0);
    const unsigned int gy = get_global_id(1);

    const unsigned int lx = get_local_id(0);
    const unsigned int ly = get_local_id(1);

    __local float tile[TILE_SIZE][TILE_SIZE + 1];

    
   
    tile[ly][lx] = A[gx + K * gy];
    
    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned px = get_group_id(0)  * TILE_SIZE + ly;
    unsigned py = get_group_id(1)  * TILE_SIZE + lx;

    A_t[px * K + py] = tile[lx][ly];
}