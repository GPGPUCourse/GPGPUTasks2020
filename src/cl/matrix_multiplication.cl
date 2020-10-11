#define TILE_SIZE 16

__kernel void matrix_multiplication(__global float *as, __global float *bs, __global float *cs,
                                    unsigned int M, unsigned int K, unsigned int N)
{
    unsigned int x = get_global_id(0);
    unsigned int y = get_global_id(1);
    unsigned int xloc = get_local_id(0);
    unsigned int yloc = get_local_id(1);

    __local float a_tile[TILE_SIZE][TILE_SIZE];
    __local float b_tile[TILE_SIZE][TILE_SIZE];

    float result = 0;
    for (unsigned int zblock = 0; zblock < K; zblock += TILE_SIZE) {
        barrier(CLK_LOCAL_MEM_FENCE);
        unsigned int z;
        z = zblock + xloc;
        a_tile[yloc][xloc] = (y < M && z < K) ? as[y * K + z] : 0;
        z = zblock + yloc;
        b_tile[yloc][xloc] = (x < N && z < K) ? bs[z * N + x] : 0;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (unsigned int zloc = 0; zloc < TILE_SIZE; ++zloc) {
            result += a_tile[yloc][zloc] * b_tile[zloc][xloc];
        }
    }
    if (x < N && y < M)
        cs[y * N + x] = result;
}