#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void merge(__global float *as, __global float *swap_array,
                    const unsigned int n, const unsigned int merge_size) {
    // сливаем два массива размера merge_size

    unsigned int global_id = get_global_id(0);

    if (global_id >= n) {
        return;
    }

    int merge_pair_size = (merge_size << 1);
    int merge_id = global_id / merge_pair_size;    // id сливаемого подмассива, частью которого является workitem

    int left_border = merge_pair_size * merge_id;  // начало пары сливаемых массивов
    int half_border = left_border + merge_size;    // середина пары сливаемых массивов

    if (half_border >= n) {
        swap_array[global_id] = as[global_id];
        return;
    }

    int diag_id = global_id - left_border;
    bool is_upper_diag = (diag_id <= merge_size);
    int diag_size = is_upper_diag ? diag_id : merge_pair_size - diag_id;

    // бинпоиск по диагонали workitem-а
    int lower_bound = 0;
    int upper_bound = diag_size;

    while (lower_bound < upper_bound) {
        int mid = lower_bound + (upper_bound - lower_bound) / 2;

        int a_id = is_upper_diag ? (diag_id - 1) - mid : (merge_size - 1) - mid;
        int b_id = is_upper_diag ? mid : merge_size - diag_size + mid;

        if (a_id >= merge_size || as[left_border + a_id] >= as[half_border + b_id]){
            lower_bound = mid + 1;
        } else{
            upper_bound = mid;
        }
    }

    int a_id = is_upper_diag ? diag_id - lower_bound : merge_size - lower_bound;
    int b_id = is_upper_diag ? lower_bound : merge_size - diag_size + lower_bound;

    if (b_id < merge_size && (a_id >= merge_size || as[left_border + a_id] > as[half_border + b_id])){
        swap_array[global_id] = as[half_border + b_id];
    } else {
        swap_array[global_id] = as[left_border + a_id];
    }
}