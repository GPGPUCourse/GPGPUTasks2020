#define TILE_SIZE 16

__kernel void matrix_transpose(__global float* a, __global float* a_t, unsigned int m, unsigned int k)
{
    //to save my brain
    int i = get_global_id(1);
    int j = get_global_id(0);

    if (i * k + j >= m * k) {
        return;
    }

    __local float tile[TILE_SIZE][TILE_SIZE];

    int local_i = get_local_id(1);
    int local_j = get_local_id(0);

    tile[local_i][local_j] = a[i * k + j];

    barrier(CLK_LOCAL_MEM_FENCE);

    float element_to_write = tile[local_j][local_i];

//    float tmp = tile[local_j][local_i];
//    tile[local_j][local_i] = tile[local_i][local_j];
//    barrier(CLK_LOCAL_MEM_FENCE);
//    tile[local_i][local_j] = tmp;

    int new_i = get_group_id(1) * TILE_SIZE + local_j;
    int new_j = get_group_id(0) * TILE_SIZE + local_i;

    a_t[new_j * m + new_i] = element_to_write;

}