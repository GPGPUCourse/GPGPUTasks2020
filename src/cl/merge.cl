#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

#define WORK_GROUP_SIZE 128

__kernel void merge(__global const float *in, __global float *out, int len, int n) {

    int id = get_global_id(0);
    if (id >= n) {
        return;
    }

    int left_block_start = id / (2 * len) * 2 * len;

    if (left_block_start != id) {
        return;
    }

    int right_block_start = left_block_start + len;

    if (right_block_start >= n) {
        for (int i = left_block_start; i < n; ++i) {
            out[i] = in[i];
        }
        return;
    }
    if (len == 1) {
        out[left_block_start] = min(in[left_block_start], in[left_block_start + 1]);
        out[left_block_start + 1] = max(in[left_block_start], in[left_block_start + 1]);
        return;
    }

    int pout = left_block_start;
    int pl = left_block_start;
    int pr = pl + len;
    int right_bound = (left_block_start + 2 * len < n) ? (left_block_start + 2 * len) : (n);
    for (pout = left_block_start; pout < right_bound; ++pout) {
        if (pr >= right_bound) {
            out[pout] = in[pl];
            ++pl;
            continue;
        }
        if (pl >= left_block_start + len) {
            out[pout] = in[pr];
            ++pr;
            continue;
        }
        if (in[pl] < in[pr]) {
            out[pout] = in[pl];
            ++pl;
        } else {
            out[pout] = in[pr];
            ++pr;
        }
    }

    return;
}