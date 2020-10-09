#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 32
__kernel void matrix_transpose(__global const float* a,
                               __global       float* a_t,
                               unsigned int M,
                               unsigned int K)
{
    const unsigned int global_id_x = get_global_id(0);
    const unsigned int global_id_y = get_global_id(1);

    const unsigned int group_id_x = get_group_id(0);
    const unsigned int group_id_y = get_group_id(1);

    const unsigned int local_id_x = get_local_id(0);
    const unsigned int local_id_y = get_local_id(1);

    __local float square[WORK_GROUP_SIZE][WORK_GROUP_SIZE + 1];
    if (global_id_x < M && global_id_y < K) {
        square[local_id_y][local_id_x] = a[global_id_y * M + global_id_x];
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    if (global_id_x < M && global_id_y < K) {
        a_t[(WORK_GROUP_SIZE * group_id_x + local_id_y) * K + (WORK_GROUP_SIZE * group_id_y + local_id_x)] = square[local_id_x][local_id_y];
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
}