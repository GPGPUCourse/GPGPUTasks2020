#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define uint unsigned int
#define cuint const uint
#define WG_SIZE 256

__kernel void merge_local(__global const float* xs,
                          __global float* res,
                          cuint n, cuint step)
{
    cuint global_id=get_global_id(0);
    cuint local_id=get_local_id(0);
    
    __local float local_xs[WG_SIZE];
    local_xs[local_id]=xs[global_id];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    cuint y_left=local_id/(2*step)*(2*step);
    cuint y_right=y_left+step;
    cuint x_left=y_right;
    cuint x_right=x_left+step;
    
    cuint y_pos=local_id-y_left;
    
    int delta=y_pos-step;
    int offset=x_left+y_pos-1;
    int l=max(delta,0)-1;
    int r=min(y_pos,step);
    int m;
    while(l+1<r)
    {
        m=(l+r)/2;
        if(local_xs[y_left+m]<=local_xs[offset-m])
            l=m;
        else
            r=m;
    }
    
    cuint y=y_left+r;
    cuint x=x_left+y_pos-r;
    const bool flag_pos=(y<y_right && (x>=x_right || local_xs[y]<=local_xs[x]));
    res[global_id]=local_xs[flag_pos?y:x];
}

__kernel void merge_init(__global const float* xs,
                         __global float* bi_y_l,
                         __global float* bi_y_r,
                         __global float* bi_x_l,
                         __global float* bi_x_r,
                         cuint n, cuint step)
{
    cuint global_id=get_global_id(0);
    if(global_id==0)
    {
        bi_y_l[0]=0;
        bi_x_l[0]=step;
        cuint wg_count=n/WG_SIZE;
        bi_y_r[wg_count-1]=n-step;
        bi_x_r[wg_count-1]=n;
        return;
    }
    
    cuint block_step=2*step/WG_SIZE;
    if(global_id%block_step==0)
    {
        cuint block_id=global_id/block_step;
        cuint x=block_id*2*step;
        bi_y_l[global_id]=x;
        bi_x_l[global_id]=x+step;
        bi_y_r[global_id-1]=x-step;
        bi_x_r[global_id-1]=x;
        return;
    }
    
    cuint y_left=(global_id*WG_SIZE)/(2*step)*(2*step);
    cuint y_right=y_left+step;
    cuint x_left=y_right;
    cuint x_right=x_left+step;
    
    cuint y_pos=(global_id*WG_SIZE)-y_left;
    
    int delta=y_pos-step;
    int offset=x_left+y_pos-1;
    int l=max(delta,0)-1;
    int r=min(y_pos,step);
    int m;
    while(l+1<r)
    {
        m=(l+r)/2;
        if(xs[y_left+m]<=xs[offset-m])
            l=m;
        else
            r=m;
    }
    
    cuint y=min(y_left+r,y_right);
    cuint x=min(x_left+y_pos-r,x_right);
    bi_y_l[global_id]=y;
    bi_x_l[global_id]=x;
    bi_y_r[global_id-1]=y;
    bi_x_r[global_id-1]=x;
}

__kernel void merge_global(__global const float* xs,
                           __global float* res,
                           __global const float* bi_y_l,
                           __global const float* bi_y_r,
                           __global const float* bi_x_l,
                           __global const float* bi_x_r,
                           cuint n, cuint step)
{
    cuint global_id=get_global_id(0);
    cuint group_id=get_group_id(0);
    cuint local_id=get_local_id(0);
    
    cuint y_left=bi_y_l[group_id];
    cuint y_right=bi_y_r[group_id];
    cuint x_left=bi_x_l[group_id];
    cuint x_right=bi_x_r[group_id];
    
    cuint y_step=y_right-y_left;
    cuint x_step=x_right-x_left;
    
    __local float local_xs[WG_SIZE];
    local_xs[local_id]=xs[(local_id<y_step)?(y_left+local_id):(x_left+local_id-y_step)];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int delta=(int)local_id-x_step;
    int offset=y_step+local_id-1;
    int l=max(delta,0)-1;
    int r=min(local_id,y_step);
    int m;
    while(l+1<r)
    {
        m=(l+r)/2;
        if(local_xs[m]<=local_xs[offset-m])
            l=m;
        else
            r=m;
    }
    
    cuint y=r;
    cuint x=y_step+local_id-r;
    const bool flag_pos=(y<y_step && (x>=WG_SIZE || local_xs[y]<=local_xs[x]));
    res[global_id]=local_xs[flag_pos?y:x];
}
