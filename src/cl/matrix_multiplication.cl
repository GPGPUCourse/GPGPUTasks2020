#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define SQUARE_SIZE 32

__kernel void matrix_multiplication(__global const float* a,
                                    __global const float* b,
                                    __global float* c,
                                    unsigned int M, unsigned int K, unsigned int N)
{
    __local float cacheA[SQUARE_SIZE * SQUARE_SIZE];
    __local float cacheB[SQUARE_SIZE * SQUARE_SIZE];
    __local float cacheC[SQUARE_SIZE * SQUARE_SIZE];

    unsigned int id = get_local_id(1) * SQUARE_SIZE + get_local_id(0);
    cacheA[id] = cacheB[id] = cacheC[id] = 0;

    for (unsigned int k = 0; k < K; k += SQUARE_SIZE) {
        if (get_local_size(0) * get_group_id(0) + get_local_id(1) < M && get_local_id(0) + k < K)
            cacheA[id] = a[(get_local_size(0) * get_group_id(0) + get_local_id(1)) * K + get_local_id(0) + k];
        if (get_local_id(1) + k < K && get_group_id(1) * get_local_size(1) + get_local_id(0) < N)
            cacheB[id] = b[(k + get_local_id(1)) * N + get_group_id(1) * get_local_size(1) + get_local_id(0)];
        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int kk = 0; kk < SQUARE_SIZE; ++kk)
            cacheC[id] += cacheA[get_local_id(0) * SQUARE_SIZE + kk] * cacheB[kk * SQUARE_SIZE + get_local_id(1)];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (get_global_id(0) < M && get_global_id(1) < N)
        c[(get_local_size(0) * get_group_id(0) + get_local_id(1)) * N + get_local_size(1) * get_group_id(1) + get_local_id(0)] = cacheC[get_local_id(0) * SQUARE_SIZE + get_local_id(1)];
}