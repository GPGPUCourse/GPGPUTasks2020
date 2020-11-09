#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

#define WORK_GROUP_SIZE 128

int is_less(__global const float *in, int i, int j, int ls, int rs) {
    if (j < 0 || ls + j >= rs)
        return 1;
    if (i < 0 || ls + i >= rs)
        return 0;
    return in[ls + i] < in[rs + j];
}

__kernel void merge(__global const float *in, __global float *out, int len, int n) {

    int id = get_global_id(0);

    if (id >= n) {
        return;
    }

    int chunk_size = 2 * len;
    int diag_num = id % chunk_size;

    int left_block_start = id / chunk_size * chunk_size;
    int right_block_start = left_block_start + len;


    if (right_block_start >= n) {
        out[id] = in[id];
        return;
    }

    if (len == 1) {
        out[left_block_start] = min(in[left_block_start], in[left_block_start + 1]);
        out[left_block_start + 1] = max(in[left_block_start], in[left_block_start + 1]);
        return;
    }

    int pos_left = len;
    int pos_right = diag_num - len;

    if (diag_num < len) {
        pos_left = diag_num;
        pos_right = 0;
    }

    --pos_left;

    int l = -1;
    int r = l;

    if (diag_num < len) {
        r = diag_num;
    } else {
        r = chunk_size - diag_num;
    }

    int l_start = right_block_start + pos_left;
    int r_start = left_block_start + pos_right;

    while (l <= r - 2) {

        int middle = (l + r) / 2;

        if (in[r_start + middle] <= in[l_start - middle]) {
            l = middle;
        } else {
            r = middle;
        }
    }

    int rl = r_start + l;
    int ll = l_start - l;

    if (diag_num < len) {
        out[id] = min(in[rl + 1], in[ll]);
        return;
    }

    if (l + 1 == len - pos_right) {
        out[id] = in[l_start + pos_right - pos_left];
        return;
    }

    if (l >= 0) {
        out[id] = min(in[rl + 1], in[ll]);
    } else {
        out[id] = in[r_start];
    }

    return;
}

__kernel void merge_old(__global const float *in, __global float *out, int len, int n) {

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