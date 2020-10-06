#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 16
__kernel void matrix_transpose(__global const float* as,
                               __global float* as_t,
                               int M,
                               int K)
{
    const unsigned int id_x = get_global_id(1);
    const unsigned int id_y = get_global_id(0);

    const unsigned int local_id_x = get_local_id(1);
    const unsigned int local_id_y = get_local_id(0);

    __local float local_a[WORK_GROUP_SIZE + 1][WORK_GROUP_SIZE + 1];
    __local float local_at[WORK_GROUP_SIZE + 1][WORK_GROUP_SIZE + 1];

    if (id_x < M && id_y < K) {
        local_a[local_id_x][local_id_y] = as[id_x * K + id_y];
    } else {
        local_a[local_id_x][local_id_y] = 0;
    }
    local_at[local_id_y][local_id_x] = local_a[local_id_x][local_id_y];

    barrier(CLK_LOCAL_MEM_FENCE);

    const unsigned int nid_x = (id_y - local_id_y) + local_id_x;
    const unsigned int nid_y = (id_x - local_id_x) + local_id_y;

    if (nid_x < K && nid_y < M) {
        as_t[nid_x * M + nid_y] = local_at[local_id_x][local_id_y];
    }
}