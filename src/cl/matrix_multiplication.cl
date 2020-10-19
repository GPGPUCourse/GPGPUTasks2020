#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#define LOCAL_WH 16
#endif

#line 7

__kernel void matrix_multiplication(
    __global float const * aMK,
    __global float const * bKN,
    __global float * cMN,
    unsigned int M,
    unsigned int K,
    unsigned int N
)
{
    // TODO
    unsigned int n = get_global_id(0);
    unsigned int m = get_global_id(1);
    unsigned int local_n = get_local_id(0);
    unsigned int local_m = get_local_id(1);

    __local float localA[LOCAL_WH*LOCAL_WH];
    __local float localB[LOCAL_WH*LOCAL_WH];

    float res_mn = 0.0;

    for (unsigned int start_k = 0; start_k < K; start_k += LOCAL_WH) {
        localA[local_m*LOCAL_WH + local_n] = start_k + local_n < K && m < M
                                            ?aMK[m*K + start_k + local_n]
                                            :0.0f;
        localB[local_m*LOCAL_WH + local_n] = start_k + local_m < K && n < N
                                            ?bKN[(start_k + local_m)*N + n]
                                            :0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);

        //printf("a %d %d %d %d %f\n", m/LOCAL_WH*LOCAL_WH, start_k, local_m, local_n, localA[local_m*LOCAL_WH + local_n]);
        //printf("b %d %d %d %d %f\n", start_k, n/LOCAL_WH*LOCAL_WH, local_m, local_n, localB[local_m*LOCAL_WH + local_n]);

        for(unsigned int local_k = 0; local_k < LOCAL_WH; ++local_k) {
            res_mn += localA[local_m*LOCAL_WH + local_k] * localB[local_k*LOCAL_WH + local_n];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (m < M && n < N) {
        cMN[m*N + n] = res_mn;
    }
}
