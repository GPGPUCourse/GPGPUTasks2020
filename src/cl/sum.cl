#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define uint unsigned int
#define cuint const uint

#define WG_SIZE 128

__kernel void sum_1(__global cuint* xs, cuint n,
                    __global uint* res)
{
    cuint global_id=get_global_id(0);
    cuint local_id=get_local_id(0);

    __local uint local_xs[WG_SIZE];
    local_xs[local_id]=global_id<n?xs[global_id]:0;
    barrier(CLK_LOCAL_MEM_FENCE);

    if(local_id==0)
    {
        uint sum=0;
        for(uint i=0;i<WG_SIZE;i++)
            sum+=local_xs[i];
        atomic_add(res, sum);
    }
}

__kernel void sum_1_1(__global cuint* xs, cuint n,
                    __global uint* res)
{
    cuint global_id=get_global_id(0);
    cuint local_id=get_local_id(0);

    __local uint local_xs[WG_SIZE];
    local_xs[local_id]=xs[global_id];//n%WG_SIZE==0
    barrier(CLK_LOCAL_MEM_FENCE);

    if(local_id==0)
    {
        uint sum=0;
        for(uint i=0;i<WG_SIZE;i++)
            sum+=local_xs[i];
        atomic_add(res, sum);
    }
}

__kernel void sum_2(__global cuint* xs, cuint n,
                    __global uint* res)
{
    cuint global_id=get_global_id(0);
    cuint local_id=get_local_id(0);

    __local uint local_xs[WG_SIZE];
    local_xs[local_id]=global_id<n?xs[global_id]:0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(uint m=WG_SIZE;m!=1;m/=2)
    {
        if(local_id<m/2)
            local_xs[local_id]+=local_xs[local_id+m/2];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(local_id==0)
        atomic_add(res, local_xs[0]);
}

__kernel void sum_2_1(__global cuint* xs, cuint n,
                      __global uint* res)
{
    cuint global_id=get_global_id(0);
    cuint local_id=get_local_id(0);

    __local uint local_xs[WG_SIZE];
    local_xs[local_id]=xs[global_id];//n%WG_SIZE==0
    barrier(CLK_LOCAL_MEM_FENCE);

    for(uint m=WG_SIZE;m!=1;m/=2)
    {
        if(local_id<m/2)
            local_xs[local_id]+=local_xs[local_id+m/2];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(local_id==0)
        atomic_add(res, local_xs[0]);
}

__kernel void sum_2_2(__global cuint* xs, cuint n,
                      __global uint* res)
{
    cuint global_id=get_global_id(0);
    cuint local_id=get_local_id(0);
    cuint group_id=get_group_id(0);
    cuint group_size=get_local_size(0);
    
    cuint alu_id=local_id%WARP_SIZE;
    cuint warp_id=local_id/WARP_SIZE;
    cuint warp_n=(group_size+WARP_SIZE-1)/WARP_SIZE;

    __local uint local_xs[WG_SIZE];
    local_xs[local_id]=xs[global_id];//n%WG_SIZE==0
    
    for(uint m=1;m<WARP_SIZE;m*=2)
    {
        cuint left=warp_id*WARP_SIZE+2*m*alu_id;
        cuint right=left+m;
        
        local_xs[left]+=local_xs[right];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(local_id==0)
    {
        for(int i=1;i<warp_n;i++)
        {
            cuint left=0;
            cuint right=i*WARP_SIZE;
            
            local_xs[left]+=local_xs[right];
        }
        atomic_add(res, local_xs[0]);
    }
}

__kernel void sum_2_3(__global cuint* xs, cuint n,
                      __global uint* res)
{
    cuint global_id=get_global_id(0);
    cuint local_id=get_local_id(0);
    cuint group_id=get_group_id(0);
    cuint group_size=get_local_size(0);
    
    cuint alu_id=local_id%WARP_SIZE;
    cuint warp_id=local_id/WARP_SIZE;
    cuint warp_n=(group_size+WARP_SIZE-1)/WARP_SIZE;

    __local uint local_xs[WG_SIZE];
    local_xs[local_id]=xs[global_id];//n%WG_SIZE==0
    
    for(uint m=1;m<WARP_SIZE;m*=2)
    {
        cuint left=warp_id*WARP_SIZE+2*m*alu_id;
        cuint right=left+m;
        
        local_xs[left]+=local_xs[right];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for(uint m=WARP_SIZE;m<warp_n*WARP_SIZE;m*=2)
    {
        cuint left=2*m*local_id;
        cuint right=left+m;
        
        if(right<group_size)
            local_xs[left]+=local_xs[right];//bank conflict!
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if(local_id==0)
        atomic_add(res, local_xs[0]);
}

__kernel void sum_3(__global cuint* xs, cuint n,
                    __global uint* res)
{
    cuint global_id=get_global_id(0);
    cuint local_id=get_local_id(0);
    cuint group_id=get_group_id(0);
    cuint group_size=get_local_size(0);
    
    cuint alu_id=local_id%WARP_SIZE;
    cuint warp_id=local_id/WARP_SIZE;
    cuint warp_n=(group_size+WARP_SIZE-1)/WARP_SIZE;

    __local uint local_xs[WG_SIZE];
    local_xs[local_id]=global_id<n?xs[global_id]:0;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for(uint m=WARP_SIZE;m<warp_n*WARP_SIZE;m*=2)
    {
        cuint ind=2*m*warp_id+alu_id;
        if(ind<group_size-m)
            local_xs[ind]+=local_xs[ind+m];
        barrier(CLK_LOCAL_MEM_FENCE);//высокие расходы
    }

    if(local_id==0)
    {
        for(int i=1;i<WARP_SIZE;i++)
            local_xs[0]+=local_xs[i];
        res[group_id]=local_xs[0];
    }
}

__kernel void sum_3_s(__global cuint* xs, cuint n,
                    __global uint* res)
{
    cuint global_id=get_global_id(0);
    cuint local_id=get_local_id(0);
    cuint group_id=get_group_id(0);
    cuint group_size=get_local_size(0);
    
    cuint alu_id=local_id%WARP_SIZE;
    cuint warp_id=local_id/WARP_SIZE;
    cuint warp_n=(group_size+WARP_SIZE-1)/WARP_SIZE;

    __local uint local_xs[WG_SIZE];
    local_xs[local_id]=global_id<n?xs[global_id]:0;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for(uint m=WARP_SIZE;m<warp_n*WARP_SIZE;m*=2)
    {
        cuint ind=2*m*warp_id+alu_id;
        if(ind<group_size-m)
            local_xs[ind]+=local_xs[ind+m];
        barrier(CLK_LOCAL_MEM_FENCE);//высокие расходы
    }

    for(int m=WARP_SIZE/2;m!=0;m/=2)
    {
        if(local_id<m)
            local_xs[local_id]+=local_xs[local_id+m];
    }

    if(local_id==0)
        res[group_id]=local_xs[0];
}

__kernel void sum_3_1(__global cuint* xs, cuint n,
                      __global uint* res)
{
    cuint global_id=get_global_id(0);
    cuint local_id=get_local_id(0);
    cuint group_id=get_group_id(0);
    cuint group_size=get_local_size(0);
    
    cuint alu_id=local_id%WARP_SIZE;
    cuint warp_id=local_id/WARP_SIZE;
    cuint warp_n=(group_size+WARP_SIZE-1)/WARP_SIZE;

    __local uint local_xs[WG_SIZE];
    local_xs[local_id]=global_id<n?xs[global_id]:0;
    
    for(uint m=1;m<WARP_SIZE;m*=2)
    {
        cuint left=warp_id*WARP_SIZE+2*m*alu_id;
        cuint right=left+m;
        
        local_xs[left]+=local_xs[right];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for(uint m=WARP_SIZE;m<warp_n*WARP_SIZE;m*=2)
    {
        cuint left=2*m*local_id;
        cuint right=left+m;
        
        if(right<group_size)
            local_xs[left]+=local_xs[right];//bank conflict!
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if(local_id==0)
        res[group_id]=local_xs[0];
}

__kernel void sum_3_2(__global cuint* xs, cuint n,
                      __global uint* res)
{
    cuint global_id=get_global_id(0);
    cuint local_id=get_local_id(0);
    cuint group_id=get_group_id(0);
    cuint group_size=get_local_size(0);
    
    cuint alu_id=local_id%WARP_SIZE;
    cuint warp_id=local_id/WARP_SIZE;
    cuint warp_n=(group_size+WARP_SIZE-1)/WARP_SIZE;

    __local uint local_xs[WG_SIZE];
    local_xs[local_id]=global_id<n?xs[global_id]:0;
    
    for(uint m=1;m<WARP_SIZE;m*=2)
    {
        cuint left=warp_id*WARP_SIZE+2*m*alu_id;
        cuint right=left+m;
        
        local_xs[left]+=local_xs[right];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(local_id==0)
    {
        for(int i=1;i<warp_n;i++)
        {
            cuint left=0;
            cuint right=i*WARP_SIZE;
            
            local_xs[left]+=local_xs[right];
        }
        res[group_id]=local_xs[0];
    }
}