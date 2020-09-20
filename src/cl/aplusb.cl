#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

#line 8 

__kernel void aplusb(__global const float* a, __global const float* b, __global float* c, unsigned int n)
{
	unsigned int index = get_global_id(0);
	if (index >= n)
    	return;
	c[index] = a[index] + b[index];
}
