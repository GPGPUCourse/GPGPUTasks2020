#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6
// WGS = work group size
#define WGS 256

// вспомогательная функция, строит in-place из локального массива массив суффиксных сумм
// для массива, размером 8:
// на начало шага 1: |a    b|     |c         d|          |e             f|             |g                  h|
// на начало шага 2: |a (a+b)      c      (c+d)|         |e          (e+f)              g              (g+h)|
// на начало шага 3: |a (a+b) (a+b+c) (a+b+c+d)           e          (e+f)         (e+f+g)         (e+f+g+h)|
// после шага 3:      a (a+b) (a+b+c) (a+b+c+d)  (a+b+c+d+e) (a+b+c+d+e+f) (a+b+c+d+e+f+g) (a+b+c+d+e+f+g+h)
void prefix_summurize(__local unsigned int* arr,
                      const unsigned int local_index) {
    for (unsigned int power_of_two = 1; power_of_two < WGS; power_of_two *= 2) {
        unsigned int from_id = (local_index / power_of_two) * 2 * power_of_two + (power_of_two - 1);
        unsigned int to_id = from_id + (local_index % power_of_two) + 1;

        if (from_id < WGS && to_id < WGS) {
            arr[to_id] += arr[from_id];
        }
        barrier(CLK_GLOBAL_MEM_FENCE); // ждём, когда все закончат эту итерацию цикла
    }
}

__kernel void radix(__global const unsigned int* from,
                    __global       unsigned int* to,
                    __global const unsigned int* tFas,
                    const unsigned int n,
                    const unsigned int bit_number)
{
    const unsigned int global_index = get_global_id(0);
    const unsigned int local_index = get_local_id(0);
    const unsigned int group_index = get_group_id(0);

    const unsigned int n_tFas = (n + WGS - 1) / WGS;
    const unsigned int totalFalses = tFas[n_tFas - 1];

    bool b;
    __local unsigned int fs[WGS];

    if (global_index < n) {
        unsigned int b_int = (from[global_index] >> bit_number) % 2;
        b = b_int == 0 ? false : true;
        if (local_index + 1 < WGS) { // пишут все потоки, кроме последнего
            fs[local_index + 1] = 1 - b_int;
        }
        if (local_index == 0) { // первый поток заодно записывает значение в самую первую ячейку
            if(group_index > 0) {
                fs[0] = tFas[group_index - 1];
            } else {
                fs[0] = 0;
            }
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE); // ждём, когда все запишут в локальный массив

    prefix_summurize(fs, local_index);

    unsigned int t = global_index - fs[local_index] + totalFalses;

    if (global_index < n) {
        unsigned int d = b ? t : fs[local_index];
        to[d] = from[global_index];
    }
}

__kernel void totalFalses(__global const unsigned int* as,
                          __global       unsigned int* tFas,
                          const unsigned int n,
                          const unsigned int bit_number) // номер бита в числе
{
    const unsigned int global_index = get_global_id(0);
    const unsigned int local_index = get_local_id(0);
    const unsigned int group_index = get_group_id(0);

    __local unsigned int local_sum[WGS];
    if (global_index < n) {
        local_sum[local_index] = 1 - ((as[global_index] >> bit_number) % 2);
    } else {
        local_sum[local_index] = 0;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    // сумма всех элементов останется в последней ячейке:
    prefix_summurize(local_sum, local_index);

    if (local_index == 0) {
        tFas[group_index] = local_sum[WGS - 1];
    }
}

__kernel void prefixSum(__global unsigned int* tFas,
                           const unsigned int n_t,
                           const unsigned int power_of_two)
{
    const unsigned int global_index = get_global_id(0);
    const unsigned int local_index = get_local_id(0);

    if (power_of_two == WGS / 2) {
        __local unsigned int local_sum[WGS];
        if (global_index < n_t) {
            local_sum[local_index] = tFas[global_index];
        } else {
            local_sum[local_index] = 0;
        }
        barrier(CLK_GLOBAL_MEM_FENCE); // ждём, когда все запишут в локальный массив

        prefix_summurize(local_sum, local_index);

        if (global_index < n_t) {
            tFas[global_index] = local_sum[local_index];
        }
    } else {
        // множество потоков отображаются в from_id,
        // ровно один поток отображается в to_id
        // т.е. много чтений ячейки tFas[from_id]
        // и ровно одна запись в ячейку tFas[to_id] => нет гонки
        unsigned int from_id = (global_index / power_of_two) * 2 * power_of_two + (power_of_two - 1);
        unsigned int to_id = from_id + (global_index % power_of_two) + 1;

        if (from_id < n_t && to_id < n_t) {
            tFas[to_id] += tFas[from_id];
        }
    }

}