#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 32
__kernel void matrix_multiplication(__global const float* a,
                                    __global const float* b,
                                    __global       float* c,
                                    unsigned int M,
                                    unsigned int K,
                                    unsigned int N)
{
    const unsigned int global_id_x = get_global_id(0);
    const unsigned int global_id_y = get_global_id(1);

    const unsigned int local_id_x = get_local_id(0);
    const unsigned int local_id_y = get_local_id(1);

    __local float a_l[WORK_GROUP_SIZE][WORK_GROUP_SIZE + 1];
    __local float b_l[WORK_GROUP_SIZE][WORK_GROUP_SIZE + 1];

    float acc = 0.0;
    for (int k_iter = 0; k_iter * WORK_GROUP_SIZE < K; ++k_iter) {
        a_l[local_id_y][local_id_x] = a[global_id_y * K + (k_iter * WORK_GROUP_SIZE + local_id_y)];
        b_l[local_id_y][local_id_x] = a[global_id_y * N + (k_iter * WORK_GROUP_SIZE + local_id_y)];
        barrier(CLK_GLOBAL_MEM_FENCE);
        for (int iter = 0; iter < WORK_GROUP_SIZE; ++iter) {
            acc += a_l[local_id_y][iter] * b_l[iter][local_id_x];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    c[global_id_y * N + global_id_x];
}