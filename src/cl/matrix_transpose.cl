#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#ifndef TILE_SIZE
    #define TILE_SIZE 16
#endif
__kernel void matrix_transpose(__global const float *matrix, __global float *matrix_t,
                               const unsigned int m, const unsigned int k) { // matrix with size m x k

    const unsigned int col_idx = get_global_id(0); // work_item id in original matrix
    const unsigned int row_idx = get_global_id(1);

    __local float tile[TILE_SIZE][TILE_SIZE + 1]; // optimizing vs. bank-conflict

    const unsigned int local_col_idx = get_local_id(0); // work_item id in tile
    const unsigned int local_row_idx = get_local_id(1);

    if (row_idx < m && col_idx < k)
        tile[local_row_idx][local_col_idx] = matrix[row_idx * k + col_idx];
    barrier(CLK_LOCAL_MEM_FENCE); // Prevents write after read

    if (row_idx < m && col_idx < k) {
        matrix_t[local_col_idx >= local_row_idx ?
                 m * (col_idx - (local_col_idx - local_row_idx)) + row_idx + (local_col_idx - local_row_idx) :
                 m * (col_idx + (local_row_idx - local_col_idx)) + row_idx - (local_row_idx - local_col_idx)
                 ] = tile[local_col_idx][local_row_idx];
    }
}