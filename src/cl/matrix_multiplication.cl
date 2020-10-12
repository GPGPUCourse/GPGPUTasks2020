#define TILE_SIZE 16

__kernel void matrix_multiplication(__global float* a, __global float* b, __global float* c,
                                    unsigned int M, unsigned int K, unsigned int N)
{
    int i = get_global_id(1);
    int j = get_global_id(0);

    if (i * N + j >= N * M) {
        return;
    }

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    int local_i = get_local_id(1);
    int local_j = get_local_id(0);

    float c_val = 0.0f;

    for (int tileK = 0; tileK * TILE_SIZE < K; tileK++) {
        tileA[local_i][local_j] = a[i * K + (tileK * TILE_SIZE + local_j)];
        tileB[local_i][local_j] = b[(tileK * TILE_SIZE + local_i) * N + j];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; k++) {
            c_val += tileA[local_i][k] * tileB[k][local_j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    c[i * N + j] = c_val;
}