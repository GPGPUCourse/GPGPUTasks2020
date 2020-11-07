#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif

// Обозначения частично взяты из https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda
// The Parallel Radix Sort Algorithm - Phase 1: Setup and Tabulation

#define WORK_GROUP_SIZE 128
#define DATA_PER_WORKITEM 64
#define L 4
#define MASK (1 << 4) - 1

__kernel void radix_counter(__global const unsigned int *as, __global unsigned int *count_array, const unsigned int n,
                            const unsigned int pass_num) {
    const unsigned int work_group_id = get_group_id(0);
    const unsigned int local_id = get_local_id(0);
    const unsigned int work_group_num = get_num_groups(0);

    // Храним подмассив исходного массива
    __local unsigned int local_array[WORK_GROUP_SIZE * DATA_PER_WORKITEM];

    unsigned int current_work_id =  work_group_id * WORK_GROUP_SIZE * DATA_PER_WORKITEM + local_id;;
    for (unsigned int i = 0; i < DATA_PER_WORKITEM; ++i) {
        if (current_work_id >= n) {
            break;
        }
        local_array[WORK_GROUP_SIZE * i + local_id] = as[current_work_id];
        current_work_id += WORK_GROUP_SIZE;
    }
    barrier(CLK_LOCAL_MEM_FENCE); // синхронизация после чтения из VRAM

    int thread_count_array[(1 << L)] = {}; // храним 2^L локальных счетчиков для каждого workitem-а

    for (int i = 0; i < DATA_PER_WORKITEM; ++i) { // Посчет в зоне ответственности workitem-а
        unsigned int current_local_work_id = local_id * DATA_PER_WORKITEM + i;
        if (work_group_id * WORK_GROUP_SIZE * DATA_PER_WORKITEM + current_local_work_id >= n) {
            break;
        }
        unsigned int radix_idx = (MASK & (local_array[current_local_work_id] >> (L * pass_num)));
        ++thread_count_array[radix_idx];
    }

    for (unsigned int i = 0; i < (1 << L); i++) { // запись в массив частот (имеет структуру вида Radix -> WorkGroup -> WorkItem)
        count_array[work_group_num * WORK_GROUP_SIZE * i + work_group_id * WORK_GROUP_SIZE + local_id] = thread_count_array[i];
    }
}

// The Parallel Radix Sort Algorithm - Phase 2: Radix Summation
__kernel void radix_prefix_sum(__global unsigned int *count_array, __global unsigned int *prefix_offset_array,
                               const unsigned int count_array_size) {

    const unsigned int group_id = get_group_id(0);
    const unsigned int local_id = get_local_id(0);
    const unsigned int number_radices_per_wg = (1 << L) / get_num_groups(0); // одна workgroup-а обрабатывает такое число radix-ов

    __local unsigned int local_array[WORK_GROUP_SIZE]; // содержит часть счетчиков для конкретного radix-а
    __local unsigned int inner_radix_offset[1]; // накопленных локальный оффсет для конкретного radix-а
    __local unsigned int global_radix_offset[(1 << L)]; // суммарный оффест для конкретного radix-а
    inner_radix_offset[0] = 0;

    unsigned int current_work_id = group_id * number_radices_per_wg * (count_array_size / (1 << L)) + local_id;
    for (unsigned int r = 0; r < number_radices_per_wg; ++r) { // цикл по radix-ам workgroup-ы
        for (unsigned int c = 0; c < (count_array_size / (1 << L)) / WORK_GROUP_SIZE; ++c) { // цикл по подгруппам массива счетчиков для конкретного radix-а
            local_array[local_id] = count_array[current_work_id];
            barrier(CLK_LOCAL_MEM_FENCE); // синхронизация после чтения из VRAM

            unsigned int last_element = local_array[WORK_GROUP_SIZE - 1];
            if (local_id == 0) { // прибавляем локальный оффсет к первому из элементов подгруппы
                local_array[0] += inner_radix_offset[0];
            }

            // In-place up-sweep phase
            unsigned int offset = 1;
            for (unsigned int i = (WORK_GROUP_SIZE >> 1); i > 0; i >>= 1) {
                barrier(CLK_LOCAL_MEM_FENCE);
                if (local_id < i) {
                    unsigned int first = offset * (2 * local_id + 1) - 1;
                    unsigned int second = offset * (2 * local_id + 2) - 1;
                    local_array[second] += local_array[first];
                }
                // offset *= 2;
                offset <<= 1;
            }

            // In-place down-sweep phase
            if (local_id == 0) {
                local_array[WORK_GROUP_SIZE - 1] = 0;
            }

            // for (unsigned int i = 1; i < WORK_GROUP_SIZE; i *= 2) {
            for (unsigned int i = 1; i < WORK_GROUP_SIZE; i <<= 1) {
                barrier(CLK_LOCAL_MEM_FENCE);
                offset >>= 1;

                if (local_id < i) {
                    unsigned int first = offset * (2 * local_id + 1) - 1;
                    unsigned int second = offset * (2 * local_id + 2) - 1;
                    unsigned int temp = local_array[first];
                    local_array[first] = local_array[second];
                    local_array[second] += temp;
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
            if (local_id == 0){ // обновляем локальный оффсет
                local_array[0] = inner_radix_offset[0];
                inner_radix_offset[0] = local_array[WORK_GROUP_SIZE - 1] + last_element;
            }

            count_array[current_work_id] = local_array[local_id];
            current_work_id += WORK_GROUP_SIZE;
        }

        if (local_id == 0){ // записываем суммарный оффсет
            prefix_offset_array[group_id * number_radices_per_wg + r] = inner_radix_offset[0];
            inner_radix_offset[0] = 0;
        }
    }
}


// The Parallel Radix Sort Algorithm - Phase 3: Reordering
__kernel void radix_reorder(__global const unsigned int *as, __global unsigned int *swap_array,
                            __global const unsigned int *prefix_sum_array,
                            __global const unsigned int *radix_offset, const unsigned int n, const unsigned int pass_num) {
    const unsigned int work_group_id = get_group_id(0);
    const unsigned int local_id = get_local_id(0);
    const unsigned int work_group_num = get_num_groups(0);

    __local unsigned int total_radix_offset[(1 << L)]; // глобальные префиксные суммы для массива оффсетов radix-ов


    if (local_id < (1 << L)) {
        total_radix_offset[local_id] = radix_offset[local_id]; // читаем оффсеты каждой workgroup-ой
    }
    barrier(CLK_LOCAL_MEM_FENCE); // барьер после чтения из VRAM

    // calculate prefix sum for radix offsets
    if (local_id == 0){ // подсчет глобальных префиксных сумм для radix-ов (так как размер 2^L, то можно сделать одним потоком)
        unsigned int current_element = 0;
        unsigned int pref_sum = 0;
        for (unsigned int i = 0; i < (1 << L); ++i) {
            pref_sum += current_element;
            current_element = total_radix_offset[i];
            total_radix_offset[i] = pref_sum;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int count_prefix_sum_array[(1 << L)] = {}; // счетчики для каждого workitem-а
    for (unsigned int i = 0; i < (1 << L); ++i) {
        count_prefix_sum_array[i] = total_radix_offset[i] + prefix_sum_array[work_group_num * WORK_GROUP_SIZE * i + work_group_id * WORK_GROUP_SIZE + local_id];
    }


    // чтение исходного массива (как в Phase 1)
    __local unsigned int local_array[WORK_GROUP_SIZE * DATA_PER_WORKITEM];

    unsigned int current_work_id = work_group_id * WORK_GROUP_SIZE * DATA_PER_WORKITEM + local_id;
    for (unsigned int i = 0; i < DATA_PER_WORKITEM; ++i) {
        if (current_work_id >= n) {
            break;
        }
        local_array[WORK_GROUP_SIZE * i + local_id] = as[current_work_id];
        current_work_id += WORK_GROUP_SIZE;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // заполнение промежуточного массива, отсортированного по текущим L битам (not coalesced access)
    for (unsigned int i = 0; i < DATA_PER_WORKITEM; ++i) {
        unsigned int current_local_work_id = local_id * DATA_PER_WORKITEM + i;
        if (current_local_work_id >= n) {
            break;
        }
        unsigned int array_element = local_array[current_local_work_id];
        unsigned int radix_idx = (MASK & (array_element >> (L * pass_num)));

        swap_array[count_prefix_sum_array[radix_idx]] = local_array[current_local_work_id];
        ++count_prefix_sum_array[radix_idx];
    }
}

