#ifndef WORK_GROUP_SIZE
    #define WORK_GROUP_SIZE 128
#endif

#ifndef DIGITS_PER_STEP
    #define DIGITS_PER_STEP 2
    #ifndef VALUES_PER_DIGIT
        #define VALUES_PER_DIGIT (1 << DIGITS_PER_STEP)
    #endif 
#endif

#ifndef VALUES_PER_DIGIT
    #define VALUES_PER_DIGIT 4
#endif 

__kernel void local_sum(__global unsigned int* as,
                        __global unsigned int* indexes,
                        __global unsigned int* sums,
                        unsigned int original_mask, 
                        unsigned int step, 
                        unsigned int n) {

    // Вычисляем маску для нашего раунда
    unsigned int mask = (original_mask << step);
    unsigned int global_id = get_global_id(0);
    unsigned int group_id = get_group_id(0);
    unsigned int local_id = get_local_id(0);
    unsigned int group_size = get_local_size(0);
    unsigned int max_local_id = min(group_size, n - group_size * group_id);

    // Количество "размерностей" у наших массивов
    unsigned int local_arrays_dim = (VALUES_PER_DIGIT - 1);

    __local unsigned int local_as[WORK_GROUP_SIZE];

    // Локальные суммы для кадой возможной комбинации разрядов, кроме всех единиц 
    __local unsigned int local_sums[WORK_GROUP_SIZE * (VALUES_PER_DIGIT - 1)];
    unsigned int buffers[VALUES_PER_DIGIT - 1];

    if (global_id < n) {
        local_as[local_id] = as[global_id];

        // Биты с которыми мы работаем в течение шага
        // Играют роль индекса "второй размерности" в массивах сумм
        unsigned int masked = (local_as[local_id] & mask) >> step;

        if (local_id < max_local_id - 1) {
            // Забиваем локальный массив нулями
            for (int i = 0; i < local_arrays_dim; ++i) {
                local_sums[local_id + 1 + i * group_size] = 0;
            }
            // А для нужных разрядов проставляем единичку
            if (masked ^ original_mask) {
                local_sums[local_id + 1 + masked * group_size] = 1;
            }
        } else {
            for (int i = 0; i < local_arrays_dim; ++i) {
                local_sums[i * group_size] = 0;
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Вычисляем локальные префиксные суммы
    for (unsigned int k = 1; k < group_size; k *= 2) {
        if (local_id < max_local_id && local_id >= k) {
            for (int i = 0; i < local_arrays_dim; ++i) {
                buffers[i] = local_sums[local_id - k + i * group_size] + local_sums[local_id + i * group_size];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (local_id < max_local_id && local_id >= k) {
            for (int i = 0; i < local_arrays_dim; ++i) {
                local_sums[local_id + i * group_size] = buffers[i];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Складываем префиксные суммы последнего элемента подмассива
    // во внешнюю память
    if (local_id == 0) {
        unsigned int masked = (local_as[max_local_id - 1] & mask) >> step;

        unsigned int last_index = VALUES_PER_DIGIT * (group_id + 1);

        sums[last_index + local_arrays_dim] = max_local_id;

        for (int i = 0; i < local_arrays_dim; ++i) {
            unsigned int result = local_sums[max_local_id - 1 + i * group_size];
            sums[last_index + i] = result;
            sums[last_index + local_arrays_dim] -= result;
        }

        // Учитываем последний элемент подмассива (изначально был проставлен 0 выше)
        if (masked ^ original_mask) {
            sums[last_index + masked] += 1;
            sums[last_index + local_arrays_dim] -= 1;
        }
    }

    // Можем частично предподсчитать индекс (сложим его во внешнюю память)
    if (global_id < n) {
        unsigned int masked = (local_as[local_id] & mask) >> step;

        unsigned int local_index;

        if (masked ^ original_mask) {
            local_index = local_sums[local_id + masked * group_size];
        } else {
            local_index = local_id;
            for (int i = 0; i < local_arrays_dim; ++i) {
                local_index -= local_sums[local_id + i * group_size];
            }
        }

        indexes[global_id] = local_index;
    }
}

__kernel void radix(__global unsigned int* as,
                    __global unsigned int* as_result,
                    __global unsigned int* indexes,
                    __global unsigned int* sums,
                    unsigned int original_mask, 
                    unsigned int step, 
                    unsigned int n) {

    unsigned int mask = original_mask << step;
    unsigned int global_id = get_global_id(0);
    unsigned int group_id = get_group_id(0);
    unsigned int group_count = get_num_groups(0);
    unsigned int last_index = VALUES_PER_DIGIT * group_count;

    // Используя раннее частично предподсчитанный индекс и префиксные суммы
    // вычисляем новый индекс элемента
    if (global_id < n) {
        unsigned int value = as[global_id];
        unsigned int masked = (value & mask) >> step;

        unsigned int new_index = indexes[global_id] + sums[VALUES_PER_DIGIT * group_id + masked];

        for (int i = 0; i < masked; ++i) {
            new_index += sums[last_index + i];
        }

        as_result[new_index] = value;
    }
}
