#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

#line 8 // Седьмая строчка теперь восьмая (при ошибках компиляции в логе компиляции будут указаны корректные строчки благодаря этой директиве)

__kernel void aplusb(__global const float* as, __global const float* bs, __global float* cs, unsigned int n)
{
    const unsigned int index = get_global_id(0);
    if (index >= n) {
        return;
    }

    cs[index] = as[index] + bs[index];
}
