#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define GROUP_SIZE_0 16
#define GROUP_SIZE_1 8

__kernel void matrix_transpose(__global const float *a, __global float *a_t, unsigned int K, unsigned int M) {
    int g0 = get_global_id(0);
    int g1 = get_global_id(1);
    int l0 = get_local_id(0);
    int l1 = get_local_id(1);
    int j0 = get_group_id(0);
    int j1 = get_group_id(1);
    int s0 = get_local_size(0);
    int s1 = get_local_size(1);

    // g0 == j0 * s0 + l0
    // g1 == j1 * s1 + l1

    __local float buf[GROUP_SIZE_1][GROUP_SIZE_0 + 1];

    buf[l1][l0] = a[g1 * K + g0];

    barrier(CLK_LOCAL_MEM_FENCE);

    int n = l0 + l1 * s0;
    int q0 = n / s1;
    int q1 = n % s1;

    a_t[(j0 * s0 + q0) * M + (j1 * s1 + q1)] = buf[q1][q0];
}