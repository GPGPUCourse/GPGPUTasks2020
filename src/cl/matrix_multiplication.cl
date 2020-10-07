#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORKGROUP_SIZE_EXTRA 272
#define WORKGROUP_SIZE_X 16
#define WORKGROUP_SIZE_Y 16
#define OFFSET 1

__kernel void matrix_multiplication(__global const float* a, __global const float* b, __global float* c,
                                    unsigned int m, unsigned int k, unsigned int n)
{
    __local float a_memory[WORKGROUP_SIZE_EXTRA];
    __local float b_memory[WORKGROUP_SIZE_EXTRA];
    __local float c_memory[WORKGROUP_SIZE_EXTRA];

    const unsigned int i_local = get_local_id(1);
    const unsigned int j_local = get_local_id(0);
    const unsigned int i = get_group_id(1) * WORKGROUP_SIZE_Y;
    const unsigned int j = get_group_id(0) * WORKGROUP_SIZE_X;

    unsigned int start_point_a = i * k;
    unsigned int start_point_b = j;
    unsigned int start_point_c = i * n + j;

    c_memory[i_local * (WORKGROUP_SIZE_X + OFFSET) + j_local] = 0;

    for (unsigned int iter_i = 0; iter_i < k / WORKGROUP_SIZE_Y; ++iter_i) {
        a_memory[i_local * (WORKGROUP_SIZE_Y + OFFSET) + j_local] = a[start_point_a + i_local * k + j_local];
        b_memory[j_local * (WORKGROUP_SIZE_Y + OFFSET) + i_local] = b[start_point_b + i_local * n + j_local];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (unsigned int ii = 0; ii < WORKGROUP_SIZE_X; ++ii) {
            c_memory[i_local * (WORKGROUP_SIZE_Y + OFFSET) + j_local] +=
                    a_memory[i_local * (WORKGROUP_SIZE_Y + OFFSET) + ii] *
                    b_memory[j_local * (WORKGROUP_SIZE_Y + OFFSET) + ii];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        start_point_a += WORKGROUP_SIZE_X;
        start_point_b += WORKGROUP_SIZE_Y * n;
    }
    c[start_point_c + i_local * n + j_local] = c_memory[i_local * (WORKGROUP_SIZE_Y + OFFSET) + j_local];
}