#define TILE_SIZE 16

__kernel void matrix_multiplication(__global float *a, __global float *b, __global float *c,
                                    unsigned int M, unsigned int K, unsigned int N)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int l_i = get_local_id(0);
    int l_j = get_local_id(1);
    __local float tileA[TILE_SIZE][TILE_SIZE + 1];
    __local float tileB[TILE_SIZE][TILE_SIZE + 1];

    if (i < N && j < M) {
        float sum = 0.0f;
        for (int tileK = 0; tileK * TILE_SIZE < K; ++tileK) {
            tileA[l_j][l_i] = a[j * K + (tileK * TILE_SIZE + l_i)];
            tileB[l_j][l_i] = b[i + (tileK * TILE_SIZE + l_j) * N];
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int k = 0; k < TILE_SIZE; ++k) {
                sum += tileA[l_j][k] * tileB[k][l_i];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        c[j * N + i] = sum;
    }
}