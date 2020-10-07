#define WORKGROUP_SIZE_EXTRA 272
#define WORKGROUP_SIZE_X 16
#define WORKGROUP_SIZE_Y 16
#define OFFSET 1

__kernel void matrix_transpose(__global const float* m, __global float* m_t, unsigned int n, unsigned int k)
{
    __local float memory[WORKGROUP_SIZE_EXTRA];

    const unsigned int i_local = get_local_id(1);
    const unsigned int j_local = get_local_id(0);
    const unsigned int i = get_group_id(1) * WORKGROUP_SIZE_X;
    const unsigned int j = get_group_id(0) * WORKGROUP_SIZE_Y;

    unsigned int start_point_m = i * k + j;
    unsigned int start_point_m_t = j * n + i;
    memory[j_local * (WORKGROUP_SIZE_Y + OFFSET) + i_local] = m[start_point_m + i_local * k + j_local];
    barrier(CLK_LOCAL_MEM_FENCE);
    m_t[start_point_m_t + i_local * n + j_local] = memory[i_local * (WORKGROUP_SIZE_Y + OFFSET) + j_local];
}