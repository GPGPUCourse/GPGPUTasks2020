#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define uint unsigned int
#define cuint const uint

#define WG_SIZE 16

__kernel void matrix_multiplication(__global const float* as,
				    __global const float* bs,
				    __global float* cs,
				    cuint M, cuint K, cuint N)
{
    cuint global_id_x=get_global_id(0);
	cuint global_id_y=get_global_id(1);
    cuint local_id_x=get_local_id(0);
	cuint local_id_y=get_local_id(1);
	
	__local float local_as[WG_SIZE][WG_SIZE+1];
	__local float local_bs[WG_SIZE][WG_SIZE+1];
	float local_cs=0;
	
	for(int k0=0;k0<K;k0+=WG_SIZE)
	{
		local_as[local_id_y][local_id_x]=(k0+local_id_x<K && global_id_y<M)?
											as[global_id_y*K+k0+local_id_x]:0;
		local_bs[local_id_y][local_id_x]=(k0+local_id_y<K && global_id_x<N)?
											bs[(k0+local_id_y)*N+global_id_x]:0;
		barrier(CLK_LOCAL_MEM_FENCE);
		
		for(uint k=0;k<WG_SIZE;k++)
			local_cs+=local_as[local_id_y][k]*local_bs[k][local_id_x];
		
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	if(global_id_y<M && global_id_x<N)
		cs[global_id_y*N+global_id_x]=local_cs;
}
