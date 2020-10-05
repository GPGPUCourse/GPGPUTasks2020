<<<<<<< HEAD
// TODO
#define WORK_GROUP_SIZE 128
#define PER_ITEM 32

__kernel void sumAtomic(__global const unsigned int *xs,
                        unsigned int n,
                        __global unsigned int *res) {

    unsigned int localId = get_local_id(0);
    unsigned int globalId = get_global_id(0);

    __local unsigned int local_xs[WORK_GROUP_SIZE];

    local_xs[localId] = (globalId >= n) ? 0 : xs[globalId];

    barrier(CLK_LOCAL_MEM_FENCE);
    if (localId == 0) {
        unsigned int sum = 0;
        for (unsigned int i = 0; i < WORK_GROUP_SIZE; ++i) {
            sum += local_xs[i];
        }
        atomic_add(res, sum);
    }
}

__kernel void fastSum(__global const unsigned int *xs,
                      unsigned int n,
                      __global unsigned int *res) {

    const unsigned int localId = get_local_id(0);
    const unsigned int groupId = get_group_id(0);
    __local unsigned int local_xs[WORK_GROUP_SIZE];
    local_xs[localId] = 0;

    for (unsigned int i = 0; i < PER_ITEM; ++i) {
        const unsigned int totalId = groupId * WORK_GROUP_SIZE * PER_ITEM + i * WORK_GROUP_SIZE + localId;
        if (totalId < n) {
            local_xs[localId] += xs[totalId];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    for (unsigned int n_values = WORK_GROUP_SIZE; n_values > 1; n_values /= 2) {
        if (2 * localId < n_values) {
            unsigned int a = local_xs[localId];
            unsigned int b = local_xs[localId + n_values / 2];
            local_xs[localId] = a + b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (localId == 0) {
        atomic_add(res, local_xs[0]);
    }
}