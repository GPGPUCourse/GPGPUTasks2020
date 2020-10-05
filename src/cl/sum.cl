#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

const unsigned int workGroupSize = 128;

__kernel void sum(__global const unsigned* a,
    __local unsigned* b,
    unsigned int n)
{
    const unsigned int index = get_global_id(0);

    if (index >= n)
        return;

    if (index == 0) {
        // ���� �� printf ��� �� ��� if, �� printf ��������� �� ����������� ��� ���� ���������� workItems
        printf("Just example of printf usage: WARP_SIZE=%d\n", WARP_SIZE);
    }

    c[index] = a[index] + b[index];
}