#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define SQUARE_SIZE 32

__kernel void matrix_transpose(__global const float* matrix,
                                __global float* transposed_matrix,
                                unsigned int n, unsigned int m)
{
    __local float cache[SQUARE_SIZE][SQUARE_SIZE];
    unsigned int i = get_global_id(0), j = get_global_id(1);
    unsigned int y = (get_local_id(0) + get_local_id(1)) & (SQUARE_SIZE - 1);
    if (i < n && j < m)
        cache[get_local_id(1)][y] = matrix[j * n + i];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (i < n && j < m)
        transposed_matrix[(get_local_size(0) * get_group_id(0) + get_local_id(1)) * m + get_local_size(1) * get_group_id(1) + get_local_id(0)] = cache[get_local_id(0)][y];
}

// __kernel void matrix_transpose(__global const float* matrix,
//                                 __global float* transposed_matrix,
//                                 unsigned int n, unsigned int m)
// {
//     unsigned int i = get_global_id(0), j = get_global_id(1);
//     if (i < n && j < m)
//         transposed_matrix[i * m + j] = matrix[j * n + i];
// }