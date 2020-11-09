#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

int compare(__global float* as, int i, int j, int a_start, int b_start) {
    if (j < 0 || i >= b_start - a_start) {
        return 1;
    }

    if (i < 0 || j >= b_start - a_start) {
        return 0;
    }

    unsigned int first_id = a_start + i, second_id = b_start + j;
    return (as[first_id] < as[second_id]) ? 0 : 1;
}

int find_diag_i(int id, int diag_start) {
    return diag_start - id;
}

int find_diag_j(int id, int diag_start) {
    return diag_start + id;
}

__kernel void merge(__global float* as, __global float* bs, int chunk_size) {
    int global_id = get_global_id(0);

    int line_size = chunk_size / 2;
    int a_start = global_id / chunk_size * chunk_size;
    int b_start = a_start + line_size;
    int diag_pos = global_id % chunk_size;

    int start_diag_i = (diag_pos < line_size) ? diag_pos : line_size;
    int start_diag_j = (diag_pos < line_size) ? 0 : diag_pos - line_size;
    int end_diag_i = start_diag_j;
    int end_diag_j = start_diag_i;
    int diag_size = start_diag_i - end_diag_i + 1;

    int left = 0;
    int right = diag_size - 1;
    int curr = (left + right) / 2;

    int curr_diag_i = find_diag_i(curr, start_diag_i);
    int curr_diag_j = find_diag_j(curr, start_diag_j);
    int upper_flag = compare(as, curr_diag_i - 1, curr_diag_j, a_start, b_start);
    int lower_flag = compare(as, curr_diag_i, curr_diag_j - 1, a_start, b_start);

    while (!(upper_flag == 0 && lower_flag == 1)) {
        if (lower_flag == 0) {
            right = curr - 1;
        } else if (upper_flag == 1) {
            left = curr + 1;
        }
        curr = (left + right) / 2;

        curr_diag_i = find_diag_i(curr, start_diag_i);
        curr_diag_j = find_diag_j(curr, start_diag_j);
        upper_flag = compare(as, curr_diag_i - 1, curr_diag_j, a_start, b_start);
        lower_flag = compare(as, curr_diag_i, curr_diag_j - 1, a_start, b_start);
    }

    if (compare(as, curr_diag_i, curr_diag_j, a_start, b_start) == 0) {
        bs[global_id] = as[a_start + curr_diag_i];
    } else {
        bs[global_id] = as[b_start + curr_diag_j];
    }
}
