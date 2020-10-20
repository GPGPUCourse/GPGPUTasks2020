#define TILE_SIZE 16

__kernel void matrix_multiplication(
    __global const float* a,
    __global const float* b,
    __global float* c,
    int n,
    int m,
    int k) {
    const int global_i = get_global_id(0);
    const int global_j = get_global_id(1);
    const int local_i = get_local_id(0);
    const int local_j = get_local_id(1);
    __local float local_a[TILE_SIZE][TILE_SIZE + 1];
    __local float local_b[TILE_SIZE][TILE_SIZE + 1];

    float sum = 0.;
    for (int i = 0; i * TILE_SIZE < m; i++) {
        local_a[local_j][local_i] = a[global_j * m + i * TILE_SIZE + local_i];
        local_b[local_j][local_i] = b[(local_j + i * TILE_SIZE) * k + global_i];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int t = 0; t < TILE_SIZE; t++) {
            sum += local_a[local_j][t] * local_b[t][local_i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[global_i + global_j * k] = sum;
}
