#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define uint unsigned int
#define cuint const uint

#define WG_SIZE 16

__kernel void matrix_transpose(__global const float* xs,
							   __global float* xs_t,
							   cuint M, cuint K)
{
    cuint global_id_x=get_global_id(0);
	cuint global_id_y=get_global_id(1);
    cuint local_id_x=get_local_id(0);
	cuint local_id_y=get_local_id(1);

    __local float local_xs[WG_SIZE][WG_SIZE+1];
    
	if(global_id_y<M && global_id_x<K)
		local_xs[local_id_x][local_id_y]=xs[global_id_y*K+global_id_x];
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	cuint t_global_id_y=get_group_id(0)*WG_SIZE+local_id_y;
	cuint t_global_id_x=get_group_id(1)*WG_SIZE+local_id_x;
	
	if(t_global_id_y<K && t_global_id_x<M)
		xs_t[t_global_id_y*M+t_global_id_x]=local_xs[local_id_y][local_id_x];
}
/*
__kernel void matrix_transpose_2(__global const float* xs,
							     __global float* xs_t,
							     cuint m, cuint k)
{
    cuint global_id=get_global_id(0);
    cuint local_id=get_local_id(0);
    cuint group_id=get_group_id(0);
    cuint group_size=get_local_size(0);
    
    cuint alu_id=local_id%WARP_SIZE;
    cuint warp_id=local_id/WARP_SIZE;
    cuint warp_n=(group_size+WARP_SIZE-1)/WARP_SIZE;

    __local float local_xs[WG_SIZE][(WARP_SIZE+1)];
	__local float local_xs_t[WG_SIZE][(WARP_SIZE+1)];
    
	cuint i0=global_id/k+warp_id*WARP_SIZE;
	cuint j0=global_id%k;
	
	for(int i=0;i<WARP_SIZE;i++)
		local_xs[i][alu_id]=xs[(i0+i)*k+j0+alu_id];
	
	for(int i=0;i<WARP_SIZE;i++)
		local_xs_t[alu_id][i]=local_xs[i][alu_id];
		
	for(int i=0;i<WARP_SIZE;i++)
		xs_t[(j0+i)*m+i0+alu_id]=local_xs_t[i][alu_id];
}
*/