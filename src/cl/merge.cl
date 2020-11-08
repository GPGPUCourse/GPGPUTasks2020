#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void merge(__global float* a,
                    __global float* b,
                    unsigned int n,
                    unsigned int level)
{
    unsigned int id = get_global_id(0);
    if (id >= n)
        return;

    unsigned int start = (id >> (level + 1)) << (level + 1);
    unsigned int localID = id - start;
    unsigned int len = (1 << level);
    unsigned int shift1 = start;
    unsigned int shift2 = start + len;
    int l = -1;
    int r = localID;
    while (l + 1 < r){
        int m = (l + r) >> 1;
        int id1 = m;
        int id2 = localID - m - 1;
        if (id1 >= (1 << level) || (id2 < (1 << level) && a[shift1 + id1] > a[shift2 + id2]))
            r = m;
        else
            l = m;
    }

    int id1 = l + 1;
    int id2 = localID - l - 1;

    if (id1 < len && (id2 == len || a[shift1 + id1] <= a[shift2 + id2]))
        b[id] = a[shift1 + id1];
    else
        b[id] = a[shift2 + id2];
}
