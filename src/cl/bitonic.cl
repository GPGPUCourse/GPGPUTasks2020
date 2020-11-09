#define BLOCK_SIZE 128

// кернел для случая, когда не влезаем в локальную память
__kernel void bitonic_global(__global float* as, int n, int i, int local_size) {
    unsigned int idx = get_global_id(0);

    // в каком порядке сравнивать
    bool less = (idx / i) % 2 == 0;

    unsigned int fst = (idx / local_size) * 2 * local_size + (idx % local_size);
    unsigned int snd = fst + local_size;

    if (fst < n && snd < n) {

        float a = as[fst];
        float b = as[snd];

        if (a > b && less) {
            as[fst] = b;
            as[snd] = a;
        }

        if (a < b && !less) {
            as[fst] = b;
            as[snd] = a;
        }
    }
}


// кернел для случая, когда помещаемся в локальную память
__kernel void bitonic_using_local_mem(__global float* as, int n, int i, int local_size) {
    unsigned int global_idx = get_global_id(0);
    unsigned int local_idx = get_local_id(0);

    bool less = (global_idx / i) % 2 == 0;

    __local float local_as[2 * BLOCK_SIZE];

    if (2 * global_idx < n) {
        local_as[2 * local_idx] = as[2 * global_idx];
    }

    if (2 * global_idx + 1 < n) {
        local_as[2 * local_idx + 1] = as[2 * global_idx + 1];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int offset = (global_idx / get_local_size(0)) * get_local_size(0);

    // перестановки в маленьких блоках
    for (unsigned int s = local_size; s > 0; s /= 2) {
        unsigned int fst = (local_idx / s) * 2 * s + (local_idx % s);
        unsigned int snd = fst + s;
        if (fst + offset < n && snd + offset < n) {

            float a = local_as[fst];
            float b = local_as[snd];

            if (a > b && less) {
                local_as[fst] = b;
                local_as[snd] = a;
            }

            if (a < b && !less) {
                local_as[fst] = b;
                local_as[snd] = a;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (2 * global_idx < n) {
        as[2 * global_idx] = local_as[2 * local_idx];
    }
    if (2 * global_idx + 1 < n) {
        as[2 * global_idx + 1] = local_as[2 * local_idx + 1];
    }
}