#ifdef __CLION_IDE__
 #include <libgpu/opencl/cl/clion_defines.cl>
#endif

#define WORK_GROUP_SIZE 128
 void global_bitonic_swap(__global float *as, const unsigned int first_idx, const unsigned int second_idx,
                          const unsigned int n) {
    if (second_idx > n || as[second_idx] > as[first_idx]){
        return;
    }

    float tmp = as[second_idx];
    as[second_idx] = as[first_idx];
    as[first_idx] = tmp;
}

void local_bitonic_swap(__local float *as, const unsigned int first_idx, const unsigned int second_idx,
                        const unsigned int n) {
    if (second_idx > n || as[second_idx] > as[first_idx]){
        return;
    }

    float tmp = as[second_idx];
    as[second_idx] = as[first_idx];
    as[first_idx] = tmp;
}

__kernel void global_bitonic(__global float* as, const unsigned int step_global_range_size,
                             const unsigned int step_subrange_size, const unsigned int n) {

    unsigned int global_id = get_global_id(0);

    // находим пару элементов, которые переставляет workitem
    // номер блока внутри последовательности размера step_global_range_size
    unsigned int subrange_idx = (global_id / (step_subrange_size >> 1));

    // оффсет всего блока
    unsigned int subrange_offset = step_subrange_size * subrange_idx;

    // оффсет workitem-а внутри блока
    unsigned int local_subrange_offset = global_id % (step_subrange_size >> 1);

    // версия с блоками одинакового типа
    unsigned int first_idx = subrange_offset + local_subrange_offset;

    unsigned int second_idx = ((step_global_range_size == step_subrange_size) ?                     // если первая итерация для последовательности текущего размера
                               subrange_offset + step_subrange_size - local_subrange_offset - 1 :   // тогда для i-го элемента с начала блока берем в пару i-й с конца
                               first_idx + (step_subrange_size >> 1));                              // иначе берем соответствующий элемент из второй части блока

    global_bitonic_swap(as, first_idx, second_idx, n);
}

__kernel void local_bitonic(__global float* as, const unsigned int step_global_range_size,
                             const unsigned int step_subrange_size, const unsigned int n) {

    unsigned int local_id = get_local_id(0);
    unsigned int global_id = get_global_id(0);
    unsigned int group_id = get_group_id(0);

    __local float local_array[WORK_GROUP_SIZE << 1]; // каждый workitem переставляет два элемента

    if ((WORK_GROUP_SIZE << 1) * group_id + local_id + WORK_GROUP_SIZE < n){
        local_array[local_id] = as[(WORK_GROUP_SIZE << 1) * group_id + local_id];
        local_array[local_id + WORK_GROUP_SIZE] = as[(WORK_GROUP_SIZE << 1) * group_id + local_id + WORK_GROUP_SIZE];
    }

    for (unsigned int inner_subrange_size = step_subrange_size; inner_subrange_size >= 2; inner_subrange_size >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);

        unsigned int subrange_idx = (local_id / (inner_subrange_size >> 1));
        unsigned int subrange_offset = inner_subrange_size * subrange_idx;
        unsigned int local_subrange_offset = local_id % (inner_subrange_size >> 1);
        unsigned int first_idx = subrange_offset + local_subrange_offset;
        unsigned int second_idx = ((step_global_range_size == inner_subrange_size) ?
                                    subrange_offset + inner_subrange_size - local_subrange_offset - 1 :
                                    first_idx + (inner_subrange_size >> 1));

        local_bitonic_swap(local_array, first_idx, second_idx, n);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if ((WORK_GROUP_SIZE << 1) * group_id + local_id + WORK_GROUP_SIZE < n){
        as[(WORK_GROUP_SIZE << 1) * group_id + local_id] = local_array[local_id];
        as[(WORK_GROUP_SIZE << 1) * group_id + local_id + WORK_GROUP_SIZE] = local_array[local_id + WORK_GROUP_SIZE];
    }
}