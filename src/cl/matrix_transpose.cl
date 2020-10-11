#define TILE_SIZE 16

__kernel void matrix_transpose(__global const float *matrix,
                               __global float *matrix_transposed,
                               unsigned int M,
                               unsigned int K) {

    __local float tile[TILE_SIZE][TILE_SIZE + 1];

    if (get_global_id(0) < M && get_global_id(1) < K) {
        tile[get_local_id(1)][get_local_id(0)] = matrix[get_global_id(0) + get_global_id(1) * K];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (get_global_id(0) < M && get_global_id(1) < K) {
        matrix_transposed[(get_group_id(0) * TILE_SIZE + get_local_id(0)) * K +
                          (get_group_id(1) * TILE_SIZE + get_local_id(1))] = tile[get_local_id(1)][get_local_id(0)];
    }
}