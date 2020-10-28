#define WORKGROUP_SIZE 256

__kernel void bitonic_local(__global float* as2) {
    int g = get_global_id(0);
    int l = get_local_id(0);
    __local float asl2[WORKGROUP_SIZE];

    asl2[l] = as2[g];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k = 2; k <= WORKGROUP_SIZE; k *= 2) {
        int ord = ((g / k + 1) % 2) * 2 - 1;
        for (int len = k; len >= 2; len /= 2) {
            int len2 = len / 2;
            if (l % len < len2 && (asl2[l + len2] - asl2[l]) * ord < 0) {
                float t = asl2[l + len2];
                asl2[l + len2] = asl2[l];
                asl2[l] = t;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    as2[g] = asl2[l];
}

__kernel void bitonic_global(__global float* as2, int block_size, int red_block_size) {
    int g = get_global_id(0);

    int ord = ((g / block_size + 1) % 2) * 2 - 1;
    int len2 = red_block_size / 2;
    float t1 = as2[g];
    float t2 = as2[g + len2];
    if (g % red_block_size < len2 && (t2 - t1) * ord < 0) {
        as2[g + len2] = t1;
        as2[g] = t2;
    }
}

__kernel void bitonic_global_tail(__global float* as2, int block_size) {
    int g = get_global_id(0);
    int l = get_local_id(0);
    __local float asl2[WORKGROUP_SIZE];

    asl2[l] = as2[g];
    barrier(CLK_LOCAL_MEM_FENCE);

    int ord = ((g / block_size + 1) % 2) * 2 - 1;
    for (int len = WORKGROUP_SIZE; len >= 2; len /= 2) {
        int len2 = len / 2;
        if (l % len < len2 && (asl2[l + len2] - asl2[l]) * ord < 0) {
            float t = asl2[l + len2];
            asl2[l + len2] = asl2[l];
            asl2[l] = t;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    as2[g] = asl2[l];
}