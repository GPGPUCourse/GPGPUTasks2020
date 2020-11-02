#define WORK_GROUP_SIZE 256

__kernel void bitonic(__global float *as,
                      int n,
                      int swap_size,
                      int size) {

    const unsigned int id = get_global_id(0);
    // in what part of array are we?
    int sgn = 1;
    if (id % (2 * size) >= size)
        sgn = -1;

    if (id % (2 * swap_size) < swap_size && id < n) {

        const unsigned int to_swap = id + swap_size;
        // swap in case of wrong order
        if (to_swap < n && (as[id] - as[to_swap]) * sgn > 0) {

            float swap = as[id];
            as[id] = as[to_swap];
            as[to_swap] = swap;
        }
    }
}

__kernel void loc_bitonic(__global float* as,
                         int n,
                         int size) {

    const unsigned int localId = get_local_id(0);
    const unsigned int id = get_global_id(0);

    __local float local_as[WORK_GROUP_SIZE];

    if (id < n)
        local_as[localId] = as[id];

    // read barrier
    barrier(CLK_LOCAL_MEM_FENCE);

    int swap_size = WORK_GROUP_SIZE / 2;
    if (size < WORK_GROUP_SIZE)
        swap_size = size / 2;
    // in what part of array are we?
    int sgn = 1;
    if (id % (2 * size) >= size)
        sgn = -1;

    while (swap_size) {
        if (localId % (2 * swap_size) < swap_size) {

            const unsigned int to_swap = localId + swap_size;
            // swap in case of wrong order
            if ((local_as[localId] - local_as[to_swap]) * sgn > 0) {

                float swap = local_as[localId];
                local_as[localId] = local_as[to_swap];
                local_as[to_swap] = swap;
            }
        }
        // wait threads to go deeper
        barrier(CLK_LOCAL_MEM_FENCE);
        swap_size >>= 1;
    }

    as[id] = local_as[localId];
}
