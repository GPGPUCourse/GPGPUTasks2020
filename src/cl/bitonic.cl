#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#define WORKGROUPSIZE 256

void g_swap(__global float* as, unsigned int from, unsigned int to, unsigned int is_increasing) {
    if ((as[from] > as[to]) == is_increasing) {
        float t = as[from];
        as[from] = as[to];
        as[to] = t;
    }
}

void l_swap(__local float* as, unsigned int from, unsigned int to, unsigned int is_increasing) {
    if ((as[from] > as[to]) == is_increasing) {
        float t = as[from];
        as[from] = as[to];
        as[to] = t;
    }
}

void g_sort(__global float* as, unsigned int n, unsigned int step) {
    unsigned int from = get_global_id(0);
    unsigned int to = from + n / 2;

    if (from % n < n / 2) {
        unsigned int is_increasing = from % (2 * step) < step;
        g_swap(as, from, to, is_increasing);
    }
}

void l_sort(__local float* as, unsigned int n, unsigned int step) {
    unsigned int l_from = get_local_id(0);
    unsigned int from = get_global_id(0);

    for (unsigned int i = n; i >= 2; i /= 2) {
        unsigned int to = l_from + i / 2;
        if (l_from % i < i / 2) {
            unsigned int is_increasing = from % (2 * step) < step;
            l_swap(as, l_from, to, is_increasing);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

//__kernel void bitonic_local(__local float* as, unsigned int n)
//{
//    unsigned int id = get_global_id(0);
//    unsigned int l_id = get_local_id(0);
//
//    __local float l_as[WORKGROUPSIZE];
//
//    if (n > WORKGROUPSIZE) {
//        return 1;
//    }
//    l_as[l_id] = as[id];
//    barrier(CLK_LOCAL_MEM_FENCE);
//
//    for (int i = 2; i <= WORKGROUPSIZE; i *= 2) {
//        l_sort(l_as, i, i);
//    }
//    as[id] = l_as[l_id];
//}

//__kernel void bitonic(__global float* as, unsigned int n, unsigned int step)
//{
//    unsigned int id = get_global_id(0);
//    unsigned int l_id = get_local_id(0);
//
//    if (n > WORKGROUPSIZE) {
//        g_sort(as, n, step);
//    }
//    else {
//        __local float l_as[WORKGROUPSIZE];
//
//        l_as[l_id] = as[id];
//        barrier(CLK_LOCAL_MEM_FENCE);
//
//        for (int i = 2; i <= WORKGROUPSIZE; i *= 2) {
//            l_sort(l_as, i, step);
//        }
//        as[id] = l_as[l_id];
//    }
//}

__kernel void bitonic_global(__global float* as, unsigned int n, unsigned int step)
{
    unsigned int id = get_global_id(0);

    g_sort(as, n, step);
}
