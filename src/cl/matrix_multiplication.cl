#define TILE_SIZE 16

__kernel void matrix_multiplication(__global const float* a,
                                    __global const float* b,
                                    __global float* c,
                                    unsigned m,
                                    unsigned k,
                                    unsigned n)
{
    const size_t i = get_global_id(0);
    const size_t j = get_global_id(1);

    const size_t local_i = get_local_id(0);
    const size_t local_j = get_local_id(1);

    __local float tileA[TILE_SIZE][TILE_SIZE + 1];
    __local float tileB[TILE_SIZE][TILE_SIZE + 1];

    float sum = 0.0f;
    for (unsigned q = 0; q * TILE_SIZE < k; ++q) {
        tileA[local_j][local_i] = a[j * k + q * TILE_SIZE + local_i];
        tileB[local_j][local_i] = b[(local_j + q * TILE_SIZE) * n + i];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (unsigned r = 0; r < TILE_SIZE; ++r) {
            sum += tileA[local_j][r] * tileB[r][local_i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    c[j * n + i] = sum;
}