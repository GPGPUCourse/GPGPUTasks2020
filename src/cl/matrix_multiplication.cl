#define TILE_SIZE 16

__kernel void matrix_multiplication(__global const float *a, __global const float *b, __global float *c, unsigned int m,
                                    unsigned int k, unsigned int n) {
    unsigned int x = get_global_id(0);
    unsigned int y = get_global_id(1);
    unsigned int local_x = get_local_id(0);
    unsigned int local_y = get_local_id(1);
    __local float a_tile[TILE_SIZE][TILE_SIZE + 1];
    __local float b_tile[TILE_SIZE][TILE_SIZE + 1];
    float res = 0;
    for (unsigned int block = 0; block < k; block += TILE_SIZE) {
        unsigned int z = block + local_x;
        a_tile[local_y][local_x] = (y < m && z < k) ? a[y * k + z] : 0;
        z = block + local_y;
        b_tile[local_y][local_x] = (z < k && x < n) ? b[z * n + x] : 0;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (unsigned int local_z = 0; local_z < TILE_SIZE; ++local_z) {
            res += a_tile[local_y][local_z] * b_tile[local_z][local_x];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (x < n && y < m) {
        c[y * n + x] = res;
    }
}