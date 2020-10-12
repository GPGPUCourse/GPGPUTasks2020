#define TILE_SIZE 16
__kernel void matrix_multiplication(__global const float* A,
                                    __global const float* B,
                                    __global float* C,
                                    unsigned M, unsigned K,unsigned N)
{

    const unsigned int gx = get_global_id(0);
    const unsigned int gy = get_global_id(1);

    const unsigned int lx = get_local_id(0);
    const unsigned int ly = get_local_id(1);

    __local float tileA[TILE_SIZE][TILE_SIZE + 1];
    __local float tileB[TILE_SIZE][TILE_SIZE + 1];

    float sum = 0;

    int countIter = K / TILE_SIZE;

    for(int i = 0;i <  countIter;++i){


        tileA[ly][lx] = A[K * gy + TILE_SIZE * i + lx];
        tileB[ly][lx] = B[N * (i + TILE_SIZE + ly) + i];
        barrier(CLK_LOCAL_MEM_FENCE);

        for(int j = 0;j < TILE_SIZE;++j){
            sum = sum + tileA[ly][j] + tileB[j][lx];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }


    C[N * ly + lx] = sum;

}