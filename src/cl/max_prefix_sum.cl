#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 256

__kernel void max_prefix_sum_inplace(__global int *group_sums,
                                     __global int *prefix_sums,
                                     __global unsigned int *prefix_inds,
                                     unsigned int density, unsigned int len) {
    unsigned int g = get_global_id(0);
    unsigned int l = get_local_id(0);

    __local int loc_gs[WORK_GROUP_SIZE];
    __local int loc_ps[WORK_GROUP_SIZE];
    __local unsigned int loc_pi[WORK_GROUP_SIZE];
    if (g < len) {
        loc_gs[l] = group_sums[g * density];
        if (density > 1) {
            loc_ps[l] = prefix_sums[g * density];
            loc_pi[l] = prefix_inds[g * density];
        } else {
            int b = loc_gs[l] > 0;
            loc_ps[l] = b * loc_gs[l];
            loc_pi[l] = b * 1;
        }
    } else {
        loc_gs[l] = 0;
        loc_ps[l] = 0;
        loc_pi[l] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (l == 0) {
        int max_i = min(len - g, (unsigned int) WORK_GROUP_SIZE);
        int group_sum = 0;
        for (int i = 0; i < max_i; ++i) {
            group_sum += loc_gs[i];
        }

        int sum = 0;
        int max_sum = 0;
        int max_index = 0;
        for (int i = 0; i < max_i; ++i) {
            if (sum + loc_ps[i] > max_sum) {
                max_sum = sum + loc_ps[i];
                max_index = density * i + loc_pi[i];
            }
            sum += loc_gs[i];
        }

        group_sums[g * density] = group_sum;
        prefix_sums[g * density] = max_sum;
        prefix_inds[g * density] = max_index;
    }
}

__kernel void max_prefix_sum_swap(__global const int *group_sums,
                                  __global const int *prefix_sums,
                                  __global const unsigned int *prefix_inds,
                                  __global int *group_sums_out,
                                  __global int *prefix_sums_out,
                                  __global unsigned int *prefix_inds_out,
                                  unsigned int density, unsigned int len) {
    unsigned int g = get_global_id(0);
    unsigned int l = get_local_id(0);
    unsigned int j = get_group_id(0);

    __local int loc_gs[WORK_GROUP_SIZE];
    __local int loc_ps[WORK_GROUP_SIZE];
    __local unsigned int loc_pi[WORK_GROUP_SIZE];
    if (g < len) {
        loc_gs[l] = group_sums[g];
        if (density > 1) {
            loc_ps[l] = prefix_sums[g];
            loc_pi[l] = prefix_inds[g];
        } else {
            int b = loc_gs[l] > 0;
            loc_ps[l] = b * loc_gs[l];
            loc_pi[l] = b * 1;
        }
    } else {
        loc_gs[l] = 0;
        loc_ps[l] = 0;
        loc_pi[l] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (l == 0) {
        int max_i = min(len - g, (unsigned int) WORK_GROUP_SIZE);
        int group_sum = 0;
        for (int i = 0; i < max_i; ++i) {
            group_sum += loc_gs[i];
        }

        int sum = 0;
        int max_sum = 0;
        int max_index = 0;
        for (int i = 0; i < max_i; ++i) {
            if (sum + loc_ps[i] > max_sum) {
                max_sum = sum + loc_ps[i];
                max_index = density * i + loc_pi[i];
            }
            sum += loc_gs[i];
        }

        group_sums_out[j] = group_sum;
        prefix_sums_out[j] = max_sum;
        prefix_inds_out[j] = max_index;
    }
}