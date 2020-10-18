#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

#define TILE_SIZE 2
__kernel void matrix_multiplication(__global const float *a,
                                    __global const float *b,
                                    __global float *c,
                                    unsigned int M,
                                    unsigned int K,
                                    unsigned int N) {
    int i = get_global_id(0); //  column
    int j = get_global_id(1); //  row
    int from = 0;
    int local_x = get_local_id(0);
    int local_y = get_local_id(1);

    float sum = 0.0f;

    __local float a_tile[TILE_SIZE][TILE_SIZE];
    __local float b_tile[TILE_SIZE][TILE_SIZE];

    for (int k = 0; k * TILE_SIZE < K; ++k) {
        a_tile[local_x][local_y] = a[K * j + TILE_SIZE * k + local_x];
        b_tile[local_x][local_y] = b[N * (k * TILE_SIZE + local_y) + i];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int t = 0; t < TILE_SIZE; ++t)
            sum += a_tile[t][local_y] * b_tile[local_x][t];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[j * N + i] = sum;
}