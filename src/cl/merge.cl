#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void merge(__global const float* a, __global float* b, unsigned int m, unsigned int n) {
    unsigned int g = get_global_id(0);
    if (g >= n) {
        return;
    }

    int shift = g / (2 * m) * (2 * m);
    g -= shift;

    if (g == 2 * m - 1) {
        b[shift + 0] = a[shift + 0] < a[shift + m] ? a[shift + 0] : a[shift + m];
        return;
    }

    int index_shift = g < m ? 0 : g - m + 1;
    int diag_len = g < m ? g + 1 : g - 2 * index_shift + 1;
    int min_bound = 0;
    int max_bound = diag_len - 1;

    int i = min_bound + index_shift;
    int j = g - min_bound - index_shift;
    if (a[shift + i] < a[shift + m + j]) {
        i = max_bound + index_shift;
        j = g - max_bound - index_shift;
        if (a[shift + i] >= a[shift + m + j]) {
            while (max_bound - min_bound > 1) {
                int check = (max_bound + min_bound) / 2;
                i = check + index_shift;
                j = g - check - index_shift;

                if (a[shift + i] >= a[shift + m + j]) {
                    max_bound = check;
                } else {
                    min_bound = check;
                }
            }
            i = max_bound + index_shift;
            j = g - max_bound - index_shift;
        } else {
            i += 1;
            j -= 1;
        }
    }

    if (i == m) {
        b[shift + g + 1] = a[shift + m + j + 1];
    } else if (j == m - 1) {
        b[shift + g + 1] = a[shift + i];
    } else {
        b[shift + g + 1] = a[shift + i] < a[shift + m + j + 1] ? a[shift + i] : a[shift + m + j + 1];
    }
}
