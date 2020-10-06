#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define GROUP_SIZE 16

__kernel void matrix_multiplication(__global const float *a, __global const float *b, __global float *c,
                                    unsigned int M, unsigned int K, unsigned int N) {
    int g0 = get_global_id(0);
    int g1 = get_global_id(1);
    int l0 = get_local_id(0);
    int l1 = get_local_id(1);

    __local float a_q[GROUP_SIZE][GROUP_SIZE + 1];
    __local float b_q[GROUP_SIZE][GROUP_SIZE + 1];

    float sum = 0;
    for (int q = 0; q * GROUP_SIZE < N; q++) {
        a_q[l1][l0] = a[(g1) * K + (GROUP_SIZE * q + l0)];
        b_q[l1][l0] = b[(GROUP_SIZE * q + l1) * K + (g0)];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i = 0; i < GROUP_SIZE; i++) {
            sum += a_q[l1][i] * b_q[i][l0];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[g1 * M + g0] = sum;
}