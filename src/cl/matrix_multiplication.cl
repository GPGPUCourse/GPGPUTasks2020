#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6
#define TILE_SIZE 16

__kernel void matrix_multiplication(__global float* a, __global float* b, __global float* c,
                                    unsigned int m, unsigned int k, unsigned int n) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    int local_x = get_local_id(0);
    int local_y = get_local_id(1);

    int group_x = get_group_id(0);
    int group_y = get_group_id(1);

    __local float tileA[TILE_SIZE + 1][TILE_SIZE + 1];
    __local float tileB[TILE_SIZE + 1][TILE_SIZE + 1];

    float sum = 0.0f;
    for (int tileK = 0; TILE_SIZE * tileK < k; ++tileK) {
        tileA[local_y][local_x] = a[y * k + (TILE_SIZE * tileK) + local_x];
        tileB[local_y][local_x] = b[(local_y + TILE_SIZE * tileK) * n + x];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (unsigned int pos = 0; pos < TILE_SIZE; ++pos) {
            sum += tileA[local_y][pos] * tileB[pos][local_x];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    c[y * n + x] = sum;
}