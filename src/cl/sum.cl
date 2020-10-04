#define WORK_GROUP_SIZE 128
#line 3
__kernel void sum(__global const unsigned int* numbers,
                  __global unsigned int* res,
                  unsigned int n)
{
    int local_id = get_local_id(0);
    int global_id = get_global_id(0);

    __local int local_xs[WORK_GROUP_SIZE];
    if (global_id >= n) {
        local_xs[local_id] = 0;
    } else {
        local_xs[local_id] = numbers[global_id];
    }


    barrier(CLK_LOCAL_MEM_FENCE);
    for (int n_values = WORK_GROUP_SIZE; n_values > 1; n_values /= 2) {
        if (2 * local_id < n_values) {
            unsigned int a = local_xs[local_id];
            unsigned int b = local_xs[local_id + n_values / 2];
            local_xs[local_id] = a + b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_id == 0) {
        atomic_add(res, local_xs[0]);
    }
}
