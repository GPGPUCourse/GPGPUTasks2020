#define TILE_SIZE 16

__kernel void matrix_multiplication(__global const float *A,
                                    __global const float *B,
                                    __global float *C,
                                    unsigned int M,
                                    unsigned int K,
                                    unsigned int N) {

    __local float tile_A[TILE_SIZE][TILE_SIZE + 1];
    __local float tile_B[TILE_SIZE][TILE_SIZE + 1];

    float sum = 0;

    for (int i = 0; i * TILE_SIZE < K; ++i) {

        if (get_global_id(1) < M && i * TILE_SIZE + get_local_id(0) < K) {
            tile_A[get_local_id(1)][get_local_id(0)] = A[get_global_id(1) * K + TILE_SIZE * i + get_local_id(0)];
        } else {
            tile_A[get_local_id(1)][get_local_id(0)] = 0;
        }

        if (get_global_id(0) < N && i * TILE_SIZE + get_local_id(1) < K) {
            tile_B[get_local_id(1)][get_local_id(0)] = B[(TILE_SIZE * i + get_local_id(1)) * N + get_global_id(0)];
        } else {
            tile_B[get_local_id(1)][get_local_id(0)] = 0;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (get_global_id(1) < M && get_global_id(0) < N) {
            for (int j = 0; j < TILE_SIZE; ++j) {
                sum += tile_A[get_local_id(1)][j] * tile_B[j][get_local_id(0)];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (get_global_id(1) < M && get_global_id(0) < N) {
            C[get_global_id(1) * N + get_global_id(0)] = sum;
        }
    }
}