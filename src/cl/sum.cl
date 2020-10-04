#define WORK_GROUP_SIZE 256
#define VALUES_PER_WORK_ITEM 32

// Found out that this type is already defined 
// typedef unsigned int uint;

__kernel void simple_sum(__global const uint* arr, uint n, __global uint* out) {

    const uint localId = get_local_id(0);
    const uint globalId = get_global_id(0);

    __local uint localArr[WORK_GROUP_SIZE];

    // If globalId < n write arr[globalId] to localArr,
    // else write 0, because if we simply return, I guess,
    // we can get UB.
    if (globalId < n) {
        localArr[localId] = arr[globalId];
    } else {
        localArr[localId] = 0;
    }


    // Wait until all threads have finished writing values
    // to local memory
    barrier(CLK_LOCAL_MEM_FENCE);
    // Count sum of elements by the first thread
    if (localId == 0) {
        uint sum = 0;
        for (uint i = 0; i < WORK_GROUP_SIZE; ++i) {
            sum += localArr[i];
        }
        atomic_add(out, sum);
    }
}


__kernel void mass_sum(__global const uint* arr, uint n, __global uint* out) {

    const uint localId = get_local_id(0);
    const uint groupId = get_group_id(0);
    __local uint localArr[WORK_GROUP_SIZE];
    localArr[localId] = 0;

    // Store sum of VALUES_PER_WORK_ITEM count of elements
    // in localArr[localId]
    for (uint i = 0; i < VALUES_PER_WORK_ITEM; ++i) {
        const uint globalId = groupId * WORK_GROUP_SIZE * VALUES_PER_WORK_ITEM 
                              + i * WORK_GROUP_SIZE + localId;
        if (globalId < n) {
            localArr[localId] += arr[globalId];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (localId == 0) {
        uint sum = 0;
        for (uint i = 0; i < WORK_GROUP_SIZE; ++i) {
            sum += localArr[i];
        }
        atomic_add(out, sum);
    }
}


__kernel void tree_sum(__global const uint* arr, uint n, __global uint* out) {

    const uint localId = get_local_id(0);
    const uint groupId = get_group_id(0);
    __local uint localArr[WORK_GROUP_SIZE];
    localArr[localId] = 0;

    for (uint i = 0; i < VALUES_PER_WORK_ITEM; ++i) {
        const uint globalId = groupId * WORK_GROUP_SIZE * VALUES_PER_WORK_ITEM
            + i * WORK_GROUP_SIZE + localId;
        if (globalId < n) {
            localArr[localId] += arr[globalId];
        }
    }

    // Wait until all threads have finished writing values
    // to local memory
    barrier(CLK_LOCAL_MEM_FENCE);
    // Binary tree
    for (uint tree_size = WORK_GROUP_SIZE; tree_size > 1; tree_size /= 2) {
        if (2 * localId < tree_size) {
            uint a = localArr[localId];
            uint b = localArr[localId + tree_size / 2];
            localArr[localId] = a + b;
        }
        // Wait until whole "layer" is ready
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (localId == 0) {
        atomic_add(out, localArr[0]);
    }
}
