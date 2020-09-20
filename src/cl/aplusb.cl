#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif


__kernel void aplusb(__global const float *A, __global const float *B, __global float *C, unsigned int n)
{
	const unsigned int index = get_global_id(0);
	
	C[index] = A[index] + B[index];
}
