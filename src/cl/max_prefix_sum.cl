#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define cint const int
#define uint unsigned int
#define cuint const uint

#define WG_SIZE 128
 
__kernel void init(__global cint* xs, cuint n,
                   __global int* sum, __global int* result, __global int* max_sum)
{
    cuint global_id=get_global_id(0);
    
    if(global_id<n)
    {
        int a=xs[global_id];
        sum[global_id]=a;
        result[global_id]=global_id+1;
        max_sum[global_id]=a;
    }
}

__kernel void max_prefix_sum(__global cint* psum, __global cint* presult, __global cint* pmax_sum,
                             cuint n,
                             __global int* sum, __global int* result, __global int* max_sum)
{
    cuint global_id=get_global_id(0);
    cuint local_id=get_local_id(0);
    cuint group_id=get_group_id(0);
    cuint group_size=get_local_size(0);
    
    cuint alu_id=local_id%WARP_SIZE;
    cuint warp_id=local_id/WARP_SIZE;
    cuint warp_n=(group_size+WARP_SIZE-1)/WARP_SIZE;

    __local int local_psum[WG_SIZE];
    __local int local_presult[WG_SIZE];
    __local int local_pmax_sum[WG_SIZE];
    local_psum[local_id]=global_id<n?psum[global_id]:0;
    local_presult[local_id]=global_id<n?presult[global_id]:0;
    local_pmax_sum[local_id]=global_id<n?pmax_sum[global_id]:0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(uint m=1;m<WARP_SIZE;m*=2)
    {
        cuint left=warp_id*WARP_SIZE+2*m*alu_id;
        cuint right=left+m;
        
        if(2*m*alu_id+m<WARP_SIZE && right<group_size)
        {
            if(local_psum[left]+local_pmax_sum[right]>local_pmax_sum[left])
            {
                local_pmax_sum[left]=local_psum[left]+local_pmax_sum[right];
                local_presult[left]=local_presult[right];
            }
            local_psum[left]+=local_psum[right];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for(uint m=WARP_SIZE;m<warp_n*WARP_SIZE;m*=2)
    {
        cuint left=2*m*local_id;
        cuint right=left+m;
        
        if(right<group_size)
        {
            if(local_psum[left]+local_pmax_sum[right]>local_pmax_sum[left])
            {
                local_pmax_sum[left]=local_psum[left]+local_pmax_sum[right];
                local_presult[left]=local_presult[right];
            }
            local_psum[left]+=local_psum[right];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if(local_id==0)
    {
        sum[group_id]=local_psum[0];
        result[group_id]=local_presult[0];
        max_sum[group_id]=local_pmax_sum[0];
    }
}

__kernel void max_prefix_sum_2(__global cint* psum, __global cint* presult, __global cint* pmax_sum,
                               cuint n,
                               __global int* sum, __global int* result, __global int* max_sum)
{
    cuint global_id=get_global_id(0);
    cuint local_id=get_local_id(0);
    cuint group_id=get_group_id(0);
    cuint group_size=get_local_size(0);
    
    cuint alu_id=local_id%WARP_SIZE;
    cuint warp_id=local_id/WARP_SIZE;
    cuint warp_n=(group_size+WARP_SIZE-1)/WARP_SIZE;

    __local int local_psum[WG_SIZE];
    __local int local_presult[WG_SIZE];
    __local int local_pmax_sum[WG_SIZE];
    local_psum[local_id]=global_id<n?psum[global_id]:0;
    local_presult[local_id]=global_id<n?presult[global_id]:0;
    local_pmax_sum[local_id]=global_id<n?pmax_sum[global_id]:0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(uint m=1;m<WARP_SIZE;m*=2)
    {
        cuint left=warp_id*WARP_SIZE+2*m*alu_id;
        cuint right=left+m;
        
        if(2*m*alu_id+m<WARP_SIZE && right<group_size)
        {
            if(local_psum[left]+local_pmax_sum[right]>local_pmax_sum[left])
            {
                local_pmax_sum[left]=local_psum[left]+local_pmax_sum[right];
                local_presult[left]=local_presult[right];
            }
            local_psum[left]+=local_psum[right];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(local_id==0)
    {
        for(int i=1;i<warp_n;i++)
        {
            cuint left=0;
            cuint right=i*WARP_SIZE;
            
            if(right<group_size)
            {
                if(local_psum[left]+local_pmax_sum[right]>local_pmax_sum[left])
                {
                    local_pmax_sum[left]=local_psum[left]+local_pmax_sum[right];
                    local_presult[left]=local_presult[right];
                }
                local_psum[left]+=local_psum[right];
            }
        }
        sum[group_id]=local_psum[0];
        result[group_id]=local_presult[0];
        max_sum[group_id]=local_pmax_sum[0];
    }
}
