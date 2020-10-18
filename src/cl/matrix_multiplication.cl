#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define TILE_SIZE 16
__kernel void naive_matrix_multiplication(__global const float* a, __global const float* b, __global float* c,
                                          unsigned int M, unsigned int K, unsigned int N)
{
    int global_i = get_global_id(0);
    int global_j = get_global_id(1);

    if (global_i < N && global_j < M)
    {
        float sum = 0;
        for (int k = 0; k < K; ++k)
            sum += a[global_j * K + k] * b[k * N + global_i];
        c[global_j * N + global_i] = sum;
    }
}

__kernel void local_memory_matrix_multiplication(__global const float* a, __global const float* b, __global float* c,
                                                 unsigned int M, unsigned int K, unsigned int N)
{
    int global_i = get_global_id(0);
    int global_j = get_global_id(1);
    
    __local float tileA[(TILE_SIZE + 1) * TILE_SIZE];
    __local float tileB[(TILE_SIZE + 1) * TILE_SIZE];
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    float sum = 0;
    for (int tileK = 0; tileK * TILE_SIZE < K; ++tileK)
    {
        if (global_i < N && global_j < M)
        {
            tileA[local_j * (TILE_SIZE + 1) + local_i] = a[global_j * K + tileK * TILE_SIZE + local_i]; 
            tileB[local_j * (TILE_SIZE + 1) + local_i] = b[(tileK * TILE_SIZE + local_j) * N + global_i]; 
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (global_i < N && global_j < M)
        {
            for (int k = 0; k < TILE_SIZE; ++k)
                sum += tileA[local_j * (TILE_SIZE + 1) + k] * tileB[k * (TILE_SIZE + 1) + local_i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    c[global_j * N + global_i] = sum;
}