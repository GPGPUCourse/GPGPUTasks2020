#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#ifndef TILE_SIZE
    #define TILE_SIZE 16
#endif
__kernel void matrix_multiplication(__global const float *matrix_first, __global const float *matrix_second,
                                    __global float *matrix_result, const unsigned int m, const unsigned int k,
                                    const unsigned int n) {
    const unsigned int col_idx = get_global_id(0); // work_item id in original matrix
    const unsigned int row_idx = get_global_id(1);

    __local float tile_matrix_first[TILE_SIZE][TILE_SIZE + 1]; // optimizing vs. bank-conflict
    __local float tile_matrix_second[TILE_SIZE][TILE_SIZE + 1];

    const unsigned int local_col_idx = get_local_id(0); // work_item id in tile
    const unsigned int local_row_idx = get_local_id(1);

    if (row_idx < m && col_idx < n) {
        float sum = 0;
        for (unsigned int tile_offset = 0; tile_offset < k; tile_offset += TILE_SIZE) {

            tile_matrix_first[local_row_idx][local_col_idx] = tile_offset + local_col_idx >= k ? 0 :
                                                              matrix_first[k * row_idx + // extract correct row
                                                                           tile_offset + local_col_idx]; // extract correct column

            tile_matrix_second[local_row_idx][local_col_idx] = tile_offset + local_col_idx >= k ? 0 :
                                                               matrix_second[n * (local_row_idx + tile_offset) + // extract correct row
                                                                             col_idx]; // extract correct column
            barrier(CLK_LOCAL_MEM_FENCE); // wait for writing into the local matrices

            for (unsigned int i = 0; i < TILE_SIZE; ++i) {
                sum += tile_matrix_first[local_row_idx][i] * tile_matrix_second[i][local_col_idx];
            }
            barrier(CLK_LOCAL_MEM_FENCE); // wait for loading from the local matrices
        }
        matrix_result[row_idx * n + col_idx] = sum;
    }
}