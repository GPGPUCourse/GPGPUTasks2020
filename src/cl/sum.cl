#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif
#define WORK_GROUP_SIZE 256
#define VALUES_PER_WORK_ITEM 64

#line 8

__kernel void simple_sum(__global const unsigned int* as, unsigned int n,
                __global unsigned int* result) {
    unsigned int localId = get_local_id(0);
    unsigned int globalId = get_global_id(0);

    __local unsigned int local_as[WORK_GROUP_SIZE];

    if (globalId >= n) {
        local_as[localId] = 0;
    } else {
        local_as[localId] = as[globalId];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (localId == 0) {
        unsigned int sum = 0;
        for (int i = 0; i < WORK_GROUP_SIZE; ++i) {
            sum += local_as[i];
        }
        atomic_add(result, sum);
    }
}

__kernel void tree_sum(__global const unsigned int* as, unsigned int n,
              __global unsigned int* result) {

    unsigned int localId = get_local_id(0);
    unsigned int groupId = get_group_id(0);

    __local unsigned int local_as[WORK_GROUP_SIZE];
    local_as[localId] = 0;

    unsigned int sum = 0;
    for (int i = 0; i < VALUES_PER_WORK_ITEM; ++i) {
        const unsigned int index = groupId * WORK_GROUP_SIZE * VALUES_PER_WORK_ITEM +
                i * WORK_GROUP_SIZE + localId;
        if (index < n) {
            local_as[localId] += as[index];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    for (int nvalues = WORK_GROUP_SIZE; nvalues > 1; nvalues /= 2) {
        if (localId * 2 < nvalues) {
            unsigned int first = local_as[localId];
            unsigned int second = local_as[localId + nvalues / 2];
            local_as[localId] = first + second;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (localId == 0) {
        atomic_add(result, local_as[0]);
    }
}