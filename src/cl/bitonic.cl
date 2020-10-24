#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6
// WGS = work group size
#define WGS 256
__kernel void bitonic(__global float* as,
                      const unsigned int n,
                      const unsigned int k,
                      const unsigned int l)
{
    const unsigned int global_id = get_global_id(0);
    const unsigned int local_id = get_local_id(0);

    bool evenness = (global_id / k) % 2 == 0 ? true : false; // true -- неубывающая, false -- невозрастающая
    if (l <= WGS) { // если можем всё победить одним вызовом
        __local float l_as[2 * WGS];
        if (2 * global_id < n) {
            l_as[2 * local_id] = as[2 * global_id];
        } else {
            l_as[2 * local_id] = INFINITY;
        }
        if (2 * global_id + 1 < n) {
            l_as[2 * local_id + 1] = as[2 * global_id + 1];
        } else {
            l_as[2 * local_id + 1] = INFINITY;
        }

        barrier(CLK_GLOBAL_MEM_FENCE); // барьер на запись в локальный массив

        for (unsigned int l_it = l; l_it > 0; l_it /= 2) {
            unsigned from = (local_id / l_it) * 2 * l_it + (local_id % l_it);
            unsigned to = from + l_it;

            if ((l_as[to] != INFINITY) && ((l_as[from] < l_as[to] && !evenness) || (l_as[from] > l_as[to] && evenness))) {
                float tmp = l_as[from];
                l_as[from] = l_as[to];
                l_as[to] = tmp;
            }
            barrier(CLK_GLOBAL_MEM_FENCE); // барьер на завершение каждой итерации l_it
        }
        if (2 * global_id < n) {
            as[2 * global_id] = l_as[2 * local_id];
        }
        if (2 * global_id + 1 < n) {
            as[2 * global_id + 1] = l_as[2 * local_id + 1];
        }
        barrier(CLK_GLOBAL_MEM_FENCE); // барьер на запись в глобальный массив
    } else { // если не можем победить всё разом
        unsigned from = (global_id / l) * 2 * l + (global_id % l);
        unsigned to = from + l;

        if ((from < n) && (to < n) && ((as[from] < as[to] && !evenness) || (as[from] > as[to] && evenness))) {
            float tmp = as[from];
            as[from] = as[to];
            as[to] = tmp;
        }
        barrier(CLK_GLOBAL_MEM_FENCE); // барьер на запись в глобальный массив
    }
}
