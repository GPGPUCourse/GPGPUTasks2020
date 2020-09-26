#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void sum_simple(__global const unsigned int* as,
                       __global unsigned int* out,
                       unsigned int n) {
    #define WORK_GROUP_SIZE 256

    size_t localId = get_local_id(0);
    size_t globalId = get_global_id(0);

    __local unsigned int local_as[WORK_GROUP_SIZE];
    if (globalId < n) {
        local_as[localId] = as[globalId];
    } else {
        local_as[localId] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (localId == 0) {
        unsigned int sum = 0;
        for (size_t i = 0; i < WORK_GROUP_SIZE; ++i) {
            sum += local_as[i];
        }
        atomic_add(out, sum);
    }
}

__kernel void sum_tree(__global const unsigned int* as,
                       __global unsigned int* out,
                       unsigned int n) {
    #define WORK_GROUP_SIZE 256

    size_t localId = get_local_id(0);
    size_t globalId = get_global_id(0);

    __local unsigned int local_as[WORK_GROUP_SIZE];
    if (globalId < n) {
        local_as[localId] = as[globalId];
    } else {
        local_as[localId] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    for (size_t nvals = WORK_GROUP_SIZE; nvals > 1; nvals /= 2) {
        if (2 * localId < nvals) {
            unsigned int a = local_as[localId];
            unsigned int b = local_as[localId + nvals/2];
            local_as[localId] = a + b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (localId == 0) {
        atomic_add(out, local_as[0]);
    }
}

__kernel void sum_mult_vals(__global const unsigned int* as,
                         __global unsigned int* out,
                         unsigned int n) {
#define WORK_GROUP_SIZE 256
#define VALUES_PER_WORK_ITEM 64

    size_t localId = get_local_id(0);
    size_t groupId = get_group_id(0);
    __local unsigned int local_as[WORK_GROUP_SIZE];
    // TODO should we zero local mem?
    local_as[localId] = 0;

    for (size_t i = 0; i < VALUES_PER_WORK_ITEM; ++i) {
        size_t as_idx = groupId * WORK_GROUP_SIZE * VALUES_PER_WORK_ITEM + i * WORK_GROUP_SIZE + localId;
        if (as_idx < n) {
            local_as[localId] += as[as_idx];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (localId == 0) {
        unsigned int sum = 0;
        for (size_t i = 0; i < WORK_GROUP_SIZE; ++i) {
            sum += local_as[i];
        }
        atomic_add(out, sum);
    }
}

__kernel void sum_tree_mult_vals(__global const unsigned int* as,
                         __global unsigned int* out,
                         unsigned int n) {
#define WORK_GROUP_SIZE 256
#define VALUES_PER_WORK_ITEM 64

    size_t localId = get_local_id(0);
    size_t groupId = get_group_id(0);
    __local unsigned int local_as[WORK_GROUP_SIZE];
    // TODO should we zero local mem?
    local_as[localId] = 0;

    for (size_t i = 0; i < VALUES_PER_WORK_ITEM; ++i) {
        size_t as_idx = groupId * WORK_GROUP_SIZE * VALUES_PER_WORK_ITEM + i * WORK_GROUP_SIZE + localId;
        if (as_idx < n) {
            local_as[localId] += as[as_idx];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    for (size_t nvals = WORK_GROUP_SIZE; nvals > 1; nvals /= 2) {
        if (2 * localId < nvals) {
            unsigned int a = local_as[localId];
            unsigned int b = local_as[localId + nvals/2];
            local_as[localId] = a + b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (localId == 0) {
        atomic_add(out, local_as[0]);
    }
}