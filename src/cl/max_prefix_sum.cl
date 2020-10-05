#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif
#define WORK_GROUP_SIZE 256
#define VALUES_PER_WORK_ITEM 2

#line 8

__kernel void prefix_sum(__global const int* in, __global int* out, unsigned int n) {
    unsigned int localId = get_local_id(0);
    unsigned int groupId = get_group_id(0);

    __local int local_sum[WORK_GROUP_SIZE];
    __local int local_prefix[WORK_GROUP_SIZE];
    unsigned int current = VALUES_PER_WORK_ITEM * localId;

    for (int i = 0; i < WORK_GROUP_SIZE; ++i) {
        local_sum[i] = 0;
        local_prefix[i] = 0;
    }

    int sum = 0;
    for (int i = 0; i < VALUES_PER_WORK_ITEM; ++i) {
        unsigned int index = i + WORK_GROUP_SIZE * groupId + current;
        if (index < n) {
            sum += in[index];
            if (sum > local_sum[localId]) {
                local_prefix[localId] = sum;
            }
            local_sum[localId] = sum;
        } else {
            local_sum[localId] = 0;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (localId == 0) {
        for (int i = 1; i < WORK_GROUP_SIZE; ++i) {
            local_prefix[0] = max(local_prefix[0], local_prefix[i] + local_sum[0]);
            local_sum[0] += local_sum[i];
        }
    }

    out[groupId] = local_prefix[0];
}