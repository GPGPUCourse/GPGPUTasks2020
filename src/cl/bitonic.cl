#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define uint unsigned int
#define cuint const uint

#define WG_SIZE 256

__kernel void bitonic_local(__global float* xs, cuint n, cuint step)
{
    cuint global_id=get_global_id(0);
	cuint group_id=get_group_id(0);
    cuint local_id=get_local_id(0);
	cuint flag=1-(global_id/step)%2;

	__local float local_xs[2*WG_SIZE];

	cuint source_ind_left=group_id*WG_SIZE*2+local_id;
	local_xs[local_id]=source_ind_left<n?xs[source_ind_left]:0;

	cuint source_ind_right=source_ind_left+WG_SIZE;
	local_xs[local_id+WG_SIZE]=source_ind_right<n?xs[source_ind_right]:0;

	barrier(CLK_LOCAL_MEM_FENCE);

	for(uint local_step=WG_SIZE;local_step!=0;local_step/=2)
	{
		cuint left=(local_id/local_step)*local_step*2+(local_id%local_step);
		cuint right=left+local_step;
		if(left<n && right<n &&
			((flag && local_xs[right]<local_xs[left]) ||
			(!flag && local_xs[right]>local_xs[left])))
		{
			float tmp=local_xs[left];
			local_xs[left]=local_xs[right];
			local_xs[right]=tmp;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if(source_ind_left<n)
		xs[source_ind_left]=local_xs[local_id];
	if(source_ind_right<n)
		xs[source_ind_right]=local_xs[local_id+WG_SIZE];
}

__kernel void bitonic_global(__global float* xs, cuint n, cuint step,cuint local_step)
{
    cuint global_id=get_global_id(0);
	cuint flag=1-(global_id/step)%2;

	cuint left=(global_id/local_step)*local_step*2+(global_id%local_step);
	cuint right=left+local_step;

	if(left<n && right<n)
	{
		float a=xs[left];
		float b=xs[right];
		if((flag && b<a) || (!flag && b>a))
		{
			xs[left]=b;
			xs[right]=a;
		}
	}
}

/*
__kernel void bitonic_global(__global float* xs, cuint n, cuint step)
{
    cuint global_id=get_global_id(0);
	cuint group_id=get_group_id(0);
    cuint local_id=get_local_id(0);
	cuint flag=1-(global_id/step)%2;

	for(uint local_step=step;local_step!=WG_SIZE;local_step/=2)
	{
		cuint left=(global_id/local_step)*local_step*2+(global_id%local_step);
		cuint right=left+local_step;

		if(left<n && right<n)
		{
			float a=xs[left];
			float b=xs[right];
			if((flag && b<a) || (!flag && b>a))
			{
				xs[left]=b;
				xs[right]=a;
			}
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}
*/
