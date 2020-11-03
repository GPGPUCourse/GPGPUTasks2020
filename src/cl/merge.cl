#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6
#define WGS 256

// Хотим слить массивы X и Y длиной x' и y' соответственно
// Пусть есть координата x, находящаяся в интервале [0, x' - 1]
//          и координата y, находящаяся в интервале [0, y' - 1]
// тогда мы можем перейти к набору координат (k, l),
// k = x + y, l = x, то есть диагональ и позиция на диагонали
// где k лежит в интервале [0, x' + y' - 1],
//         а l в интервале [max(0, k - y'), min(k, x')]
// (обратное преобразование: x = l, y = k - l)
// сделаем вид, будто X[x'] = +inf, Y[y'] = +inf

// Возвращаем позицию первого ноля, если он есть, иначе либо k + 1, либо x_n,
// т.е. первый элемент после последовательности
#define BINSEARCH { \
    if (k == 0) {\
        if (X[0] > Y[0]) {\
            return 0;\
        } else {\
            return 1;\
        }\
    }\
    if (k == x_n + y_n - 1) {\
        return x_n;\
    }\
    unsigned int l_l, l_r;\
    if (k > y_n - 1) {\
        l_l = k - y_n + 1;\
    } else {\
        l_l = 0;\
    }\
    if (k + 1 < x_n) {\
        l_r = k + 1;  \
    } else {\
        l_r = x_n;\
    } \
    for (;l_r > l_l;) {\
        unsigned int l_i = (l_l + l_r) / 2; \
        if (X[l_i] <= Y[k - l_i]) {\
            l_l = l_i + 1;\
        } else {\
            l_r = l_i;\
        }\
    }\
    return l_r;\
}

unsigned int binsearch_l(__local const float* X, unsigned int x_n,
                         __local const float* Y, unsigned int y_n,
                         const unsigned int k)
BINSEARCH

unsigned int binsearch_g(__global const float* X, unsigned int x_n,
                         __global const float* Y, unsigned int y_n,
                         const unsigned int k)
BINSEARCH

__kernel void merge(__global const float* a,
                    __global       float* b,
                    const unsigned int n, // общее число элементов
                    const unsigned int s) // размер уже упорядоченных подмассивов для s > (WGS / 2)
{
    const unsigned int glb_id = get_global_id(0);
    const unsigned int grp_id = get_group_id(0);
    const unsigned int lcl_id = get_local_id(0);

    if (s < WGS) {
        __local float a_local[WGS];
        __local float b_local[WGS];
        __local float* from = a_local;
        __local float* to = b_local;

        if (glb_id < n) {
            from[lcl_id] = a[glb_id];
        } else {
            from[lcl_id] = INFINITY;
        }

        barrier(CLK_GLOBAL_MEM_FENCE); // барьер на запись в локальный массив

        for (unsigned int s_t = 1; s_t < WGS; s_t *= 2) {
            // X[j] = a[x_i + j], Y[j] = a[y_i + j]
            const unsigned int x_i = (lcl_id) / (2 * s_t) * (2 * s_t);
            const unsigned int x_n = min(s_t, n - x_i);
            const unsigned int y_i = min(x_i + s_t, n);
            const unsigned int y_n = min(s_t, n - y_i);

            __local float* X = from + x_i;
            __local float* Y = from + y_i;

            const unsigned int k = (lcl_id) % (2 * s_t);

            unsigned int l = binsearch_l(X, x_n, Y, y_n, k);

            barrier(CLK_GLOBAL_MEM_FENCE);

            float value;

            if (l == k + 1 || y_n == 0) { //    _ верхний край и частный случай -- пустой подмассив Y
                value = X[k];             // ...1
            } else if (l == 0) { //  |0... левый край
                value = Y[k];    //   ?...
            } else if (X[l - 1] <= Y[k - l]) { // 10 единица слева от найденного ноля
                value = Y[k - l];              // 1?
            } else {              // 00 ноль слева от найденного ноля
                value = X[l - 1]; // 1?
            }

            to[lcl_id] = value;

            __local float* tmp = from;
            from = to;
            to = tmp;

            barrier(CLK_GLOBAL_MEM_FENCE);
        }
        if (glb_id < n) {
            b[glb_id] = from[lcl_id];
        }
    } else {
        // X[j] = a[x_i + j], Y[j] = a[y_i + j]
        const unsigned int x_i = (glb_id) / (2 * s) * (2 * s);
        const unsigned int x_n = min(s, n - x_i);
        const unsigned int y_i = min(x_i + s, n);
        const unsigned int y_n = min(s, n - y_i);

        __global float *X = a + x_i;
        __global float *Y = a + y_i;

        const unsigned int k = (glb_id) % (2 * s);

        if (glb_id < n) {
            unsigned int l = binsearch_g(X, x_n, Y, y_n, k);
            float value;

            if (l == k + 1 || y_n == 0) { //    _ верхний край и частный случай -- пустой подмассив Y
                value = X[k];             // ...1
            } else if (l == 0) { //  |0... левый край
                value = Y[k];    //   ?...
            } else if (X[l - 1] <= Y[k - l]) { // 10 единица слева от найденного ноля
                value = Y[k - l];              // 1?
            } else {              // 00 ноль слева от найденного ноля
                value = X[l - 1]; // 1?
            }

            b[glb_id] = value;
        }
    }
}