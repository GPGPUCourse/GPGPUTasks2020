#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 256

__kernel void merge_sort(__global float* from, __global float* to, unsigned int n, int k) {
    __local float memory[WORK_GROUP_SIZE];

    int local_id = get_local_id(0);
    int global_id = get_global_id(0);
    int bias = global_id / k * k;

//    if (k > WORK_GROUP_SIZE) {
        int i_l = max(0, global_id % k - k / 2); // 0
        int i_r = min(global_id % k, k / 2); // 1
        int i_mid = i_l + (i_r - i_l) / 2; // 0
        int j = k / 2 + global_id % k - i_mid; // 3

        while (i_l <= i_r) {
            if (from[bias + i_mid] > from[bias + j]) {
                i_r = i_mid - 1;
            } else {
                i_l = i_mid + 1;
            }
            i_mid = i_l + (i_r - i_l) / 2; // 1
            j = k / 2 + global_id % k - i_mid; // 2
        }

        i_mid = min(max(0, i_mid), k / 2 - 1); // 0
        j = k / 2 + global_id % k - i_mid; // 2

        if (j == k || from[bias + i_mid] < from[bias + j]) {
            int i_mid_prev = min(i_mid + 1, k / 2); // 2
            if (j != k / 2 && i_mid != i_mid_prev && from[bias + i_mid] < from[bias + k / 2 + global_id % k - i_mid_prev, 0]) { // true
                to[global_id] = from[bias + k / 2 + global_id % k - i_mid_prev, 0]; //
            } else {
                to[global_id] = from[bias + i_mid];
            }
//            to[global_id] = min(from[bias + k / 2 + j], from[bias + k / 2 + i_mid_prev]);
        } else {
            int i_mid_prev = max(i_mid - 1, 0);
            if (i_mid == 0 || i_mid == i_mid_prev || from[bias + i_mid_prev] < from[bias + j]) {
                to[global_id] = from[bias + j];
            } else {
                to[global_id] = from[bias + i_mid_prev];
            }
//            to[global_id] = max(from[bias + k / 2 + global_id % k - i_mid_prev], from[bias + i_mid]);
        }
//    } else {
//        if (global_id < n) {
//            memory[local_id] = from[global_id];
//        }
//
//
//    }
}
