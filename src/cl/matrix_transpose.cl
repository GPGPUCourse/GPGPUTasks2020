#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define COALESCED 1
// Должно совпадать с main_matrix_transpose.cpp и с размерами воркгруппы (LOCAL_WH,LOCAL_WH,1)
#define LOCAL_WH 16

__kernel void matrix_transpose(
    __global float const * inp, __global float * out,
    unsigned int M, unsigned int K)
{
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);
    #if !COALESCED
    if (i < K && j < M) {
        out[i*M+j] = inp[j*K + i];
    }
    #else
    unsigned int local_i = /*i % LOCAL_WH; //*/get_local_id(0);
    unsigned int local_j = /*j % LOCAL_WH; //*/get_local_id(1);
    unsigned int start_i = i / LOCAL_WH * LOCAL_WH;
    unsigned int start_j = j / LOCAL_WH * LOCAL_WH;
    __local float local_mem[LOCAL_WH*(LOCAL_WH+1)];

    unsigned int global_idx = j*K + i;
    unsigned int global_idx_out = (start_i + local_j)*M + start_j + local_i;
    unsigned int local_idx = (LOCAL_WH+1)*local_i + local_j;

    if (i < K && j < M) {
        local_mem[(LOCAL_WH+1)*local_i + local_j] = inp[global_idx];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (start_i + local_j < K && start_j + local_i < M) {
        out[global_idx_out] = local_mem[(LOCAL_WH+1)*local_j + local_i];
    }
    #endif

}
