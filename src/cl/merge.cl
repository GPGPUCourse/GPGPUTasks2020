#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 256

__kernel void merge_sort(__global float* from, __global float* to, unsigned int n, int k) {
    int local_id = get_local_id(0);
    int global_id = get_global_id(0);
    int bias = global_id / k * k;




    int i_l = max(0, global_id % k - k / 2);
    int i_r = min(global_id % k, k / 2);
    int i_mid = i_l + (i_r - i_l) / 2;
    int j = k / 2 + global_id % k - i_mid;

    while (i_l <= i_r) {
        if (j != k && from[bias + i_mid] > from[bias + j]) {
            i_r = i_mid - 1;
        } else {
            i_l = i_mid + 1;
        }
        i_mid = i_l + (i_r - i_l) / 2;
        j = k / 2 + global_id % k - i_mid;
    }

    if (j < k / 2) {
        --i_mid;
    }

    i_mid = min(max(0, i_mid), k / 2 - 1); // 1
    j = k / 2 + global_id % k - i_mid; // 7

    if (j == k || from[bias + i_mid] < from[bias + j]) {
        int i_mid_prev = min(min(i_mid + 1, global_id % k), k / 2);
        if (i_mid != i_mid_prev && from[bias + i_mid] < from[bias + k / 2 + global_id % k - i_mid_prev]) {
            to[global_id] = from[bias + k / 2 + global_id % k - i_mid_prev];
        } else {
            to[global_id] = from[bias + i_mid];
        }
    } else {
        int i_mid_prev = max(min(i_mid - 1, global_id % k), 0);
        if (i_mid == i_mid_prev || from[bias + i_mid_prev] < from[bias + j]) {
            to[global_id] = from[bias + j];
        } else {
            to[global_id] = from[bias + i_mid_prev];
        }
    }
}
