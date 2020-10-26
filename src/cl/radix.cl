#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define uint unsigned int
#define cuint const uint
#define WG_SIZE 256

void prefix_sum_local(__local uint* local_xs, cuint local_id)
{
    for(uint local_step=1;local_step<WG_SIZE;local_step*=2)
	{
        cuint left=(local_id/local_step)*local_step*2+(local_step-1);
		cuint right=left+(local_id%local_step)+1;

        if(left<WG_SIZE && right<WG_SIZE)
            local_xs[right]+=local_xs[left];

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel void prefix_local(__global cuint* xs,
						   __global uint* prefix_sums,
                           cuint n, cuint bit_pos)
{
    cuint global_id=get_global_id(0);
    cuint local_id=get_local_id(0);
    cuint group_id=get_group_id(0);

    __local uint local_prefix_sums[WG_SIZE];
	local_prefix_sums[local_id]=global_id<n?1-(xs[global_id]>>bit_pos)%2:0;
    barrier(CLK_LOCAL_MEM_FENCE);

	prefix_sum_local(local_prefix_sums,local_id);
    if(local_id==0)
        prefix_sums[group_id]=local_prefix_sums[WG_SIZE-1];
}

__kernel void prefix_global_256(__global uint* prefix_sums, cuint n)
{
    cuint global_id=get_global_id(0);
    cuint local_id=get_local_id(0);
	cuint group_id=get_group_id(0);

	__local uint local_prefix_sums[WG_SIZE];
	local_prefix_sums[local_id]=global_id<n?prefix_sums[global_id]:0;
	barrier(CLK_LOCAL_MEM_FENCE);

	prefix_sum_local(local_prefix_sums,local_id);
	if(global_id<n)
		prefix_sums[global_id]=local_prefix_sums[local_id];
}

__kernel void prefix_global(__global uint* prefix_sums, cuint n, cuint step)
{
    cuint global_id=get_global_id(0);
    cuint local_id=get_local_id(0);

	cuint left=(global_id/step)*step*2+(step-1);
	cuint right=left+(global_id%step)+1;

	if(left<n && right<n)
		prefix_sums[right]+=prefix_sums[left];
}

__kernel void radix(__global cuint* xs,
                    __global uint* res,
                    __global cuint* prefix_sums,
                    cuint n, cuint bit_pos)
{
    cuint global_id=get_global_id(0);
    cuint local_id=get_local_id(0);
    cuint group_id=get_group_id(0);

    cuint WG_COUNT=n/WG_SIZE;
    cuint zeros_count=prefix_sums[WG_COUNT-1];

    __local uint local_prefix_sums[WG_SIZE];
    uint flag=global_id<n?1-(xs[global_id]>>bit_pos)%2:0;
	if(local_id==0)
		local_prefix_sums[0]=(group_id==0?0:prefix_sums[group_id-1]);
	if(local_id!=WG_SIZE-1)
		local_prefix_sums[local_id+1]=flag;
    barrier(CLK_LOCAL_MEM_FENCE);
	if(global_id>=n)return;

    prefix_sum_local(local_prefix_sums,local_id);
    uint pos=flag?local_prefix_sums[local_id]:zeros_count+global_id-local_prefix_sums[local_id];
    res[pos]=xs[global_id];
}
/*
//optimize

//local_xs.len==2*WG_SIZE
void prefix_sum_local_2(__local uint* local_xs, cuint local_id)
{
    for(uint local_step=1;local_step<WG_SIZE;local_step*=2)
	{
        cuint left=(local_id/local_step)*local_step*2+(local_step-1);
		cuint right=left+(local_id%local_step)+1;

        if(left<2*WG_SIZE && right<2*WG_SIZE)
            local_xs[right]+=local_xs[left];

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel void prefix_local_2(__global cuint* xs,
						     __global uint* prefix_sums,
                             cuint n, cuint bit_pos)
{
    cuint global_id=get_global_id(0);
    cuint local_id=get_local_id(0);
    cuint group_id=get_group_id(0);

    __local uint local_prefix_sums[2*WG_SIZE];

	cuint source_ind_left=group_id*WG_SIZE*2+local_id;
	local_prefix_sums[local_id]=source_ind_left<n?
								1-(xs[source_ind_left]>>bit_pos)%2:0;

	cuint source_ind_right=source_ind_left+WG_SIZE;
	local_prefix_sums[local_id+WG_SIZE]=source_ind_right<n?
								1-(xs[source_ind_right]>>bit_pos)%2:0;

    barrier(CLK_LOCAL_MEM_FENCE);

	prefix_sum_local_2(local_prefix_sums,local_id);

    if(local_id==0)
        prefix_sums[group_id]=local_prefix_sums[WG_SIZE-1];
}

__kernel void prefix_global_256_2(__global uint* prefix_sums, cuint n)
{
    cuint global_id=get_global_id(0);
    cuint local_id=get_local_id(0);
	cuint group_id=get_group_id(0);

	__local uint local_prefix_sums[2*WG_SIZE];

	cuint source_ind_left=group_id*WG_SIZE*2+local_id;
	local_prefix_sums[local_id]=source_ind_left<n?prefix_sums[source_ind_left]:0;

	cuint source_ind_right=source_ind_left+WG_SIZE;
	local_prefix_sums[local_id+WG_SIZE]=source_ind_right<n?prefix_sums[source_ind_right]:0;

	barrier(CLK_LOCAL_MEM_FENCE);

	prefix_sum_local_2(local_prefix_sums,local_id);

	if(source_ind_left<n)
		prefix_sums[source_ind_left]=local_prefix_sums[local_id];
	if(source_ind_right<n)
		prefix_sums[source_ind_right]=local_prefix_sums[local_id+WG_SIZE];
}

__kernel void prefix_global_2(__global uint* prefix_sums, cuint n, cuint step)
{
    cuint global_id=get_global_id(0);
    cuint local_id=get_local_id(0);

	cuint left=(global_id/step)*step*2+(step-1);
	cuint right=left+(global_id%step)+1;

	if(left<n && right<n)
		prefix_sums[right]+=prefix_sums[left];
}

__kernel void radix_2(__global cuint* xs,
                    __global uint* res,
                    __global cuint* prefix_sums,
                    cuint n, cuint bit_pos)
{
    cuint global_id=get_global_id(0);
    cuint local_id=get_local_id(0);
    cuint group_id=get_group_id(0);

    cuint n_prefix=n/WG_SIZE/2;
    cuint zeros_count=prefix_sums[n_prefix-1];

    __local unsigned int lps[2*WG_SIZE];
	uint flag_left,flag_right;
	uint source_ind_left,source_ind_right;
	if(local_id==0)
		lps[0]=(group_id==0?0:prefix_sums[group_id-1]);
	if(local_id!=WG_SIZE-1)//err
	{
		source_ind_left=group_id*WG_SIZE*2+local_id;
		flag_left=source_ind_left<n?1-(xs[source_ind_left]>>bit_pos)%2:0;
		lps[local_id+1]=flag_left;

		source_ind_right=source_ind_left+WG_SIZE;
		flag_right=source_ind_right<n?1-(xs[source_ind_right]>>bit_pos)%2:0;
		lps[local_id+1]=flag_right;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

    prefix_sum_local_2(lps,local_id);
	if(source_ind_left<n)
	{
		uint pos=flag_left?lps[local_id]:zeros_count+source_ind_left-lps[local_id];
		res[pos]=xs[source_ind_left];
	}
	if(source_ind_right<n)
	{
		uint pos=flag_right?lps[local_id]:zeros_count+source_ind_right-lps[local_id];
		res[pos]=xs[source_ind_right];
	}
}
*/
