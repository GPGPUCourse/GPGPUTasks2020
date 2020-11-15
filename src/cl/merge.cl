#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void merge(__global const float* as, __global float* as_sorted, int n, int half_size)
{
    const int id = get_global_id(0);

    if (id >= n)
        return;

    // a - первая отсортированная половинка, b - вторая
    int a_start = (id / 2 / half_size) * 2 * half_size;
    int b_start = a_start + half_size;

    int diag = id % (2 * half_size);
    int diag_len = (diag < half_size) ? diag : 2 * half_size - diag;
    int l = -1;
    int r = diag_len;
    int shift_before_half = (diag < half_size) ? diag - 1 : half_size - 1;
    int shift_after_half = (diag < half_size) ? 0 : diag - half_size;

    while (l < r - 1) {
        int m = (l + r) / 2;
        if (as[a_start + shift_after_half + m] <= as[b_start + shift_before_half - m])
            l = m;
        else
            r = m;
    }

    int as_id;
    if (l + 1 + shift_after_half == half_size && diag >= half_size)
        as_id = b_start + shift_after_half;
    else if (l == -1 && diag >= half_size)
        as_id = a_start + shift_after_half;
    else
        as_id = (as[b_start + shift_before_half - l] < as[a_start + shift_after_half + l + 1]) ? b_start + shift_before_half - l : a_start + shift_after_half + l + 1;

    as_sorted[id] = as[as_id];
}
