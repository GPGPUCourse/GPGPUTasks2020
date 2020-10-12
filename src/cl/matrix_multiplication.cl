#define TILE_SIZE 16

__kernel void matrix_multiplication(__global const float* a,
                                    __global const float* b,
                                    __global float *c,
                                    unsigned int M,
                                    unsigned int K,
                                    unsigned int N)
{

    unsigned int gx = get_global_id(0);
    unsigned int gy = get_global_id(1);

    __local float localA[TILE_SIZE][TILE_SIZE];
    __local float localB[TILE_SIZE][TILE_SIZE];

    __local unsigned int aSize;
    aSize = M*K;
    __local unsigned int bSize;
    bSize = K*N;

    unsigned int lx = get_local_id(0);
    unsigned int ly = get_local_id(1);

    float sum = 0.0;
    for(int tileIndex = 0; tileIndex * TILE_SIZE < K; tileIndex++) {

        unsigned int aOffset = (lx + tileIndex * TILE_SIZE) + gy * K;
        unsigned int bOffset = gx + (ly + tileIndex * TILE_SIZE) * N;
//        unsigned int bOffset = (lx + tileIndex * TILE_SIZE) + (ly + tileIndex * TILE_SIZE) * N;

        float aValue = 0.0;
        if (aOffset < aSize) {
            aValue = a[aOffset];
        }

        float bValue = 0.0;
        if (bOffset < bSize) {
            bValue = b[bOffset];
        }

        localA[ly][lx] = aValue;
        localB[ly][lx] = bValue;
        barrier(CLK_LOCAL_MEM_FENCE);

        for(int k = 0; k < TILE_SIZE; k++) {
            sum += localA[ly][k] * localB[k][lx];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
//    barrier(CLK_LOCAL_MEM_FENCE);
    c[gx + gy * N] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
}