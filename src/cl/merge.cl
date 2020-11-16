#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define cint const int
#define WG_SIZE 256

__kernel void merge_local(__global const float* xs,
                          __global float* res,
                          cint n, cint step)
{
    cint global_id=get_global_id(0);
    cint local_id=get_local_id(0);
    
    __local float local_xs[WG_SIZE];
    local_xs[local_id]=xs[global_id];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    cint y_left=local_id/(2*step)*(2*step);
    cint y_right=y_left+step;
    cint x_left=y_right;
    cint x_right=x_left+step;
    
    cint y_pos=local_id-y_left;
    cint delta=y_pos-step;
    cint offset=x_left+y_pos-1;
    int l=max(delta,0)-1;
    int r=min(y_pos,step);
    int m;
    while(l+1<r)
    {
        m=(l+r)/2;
        if(local_xs[y_left+m]<local_xs[offset-m])
            l=m;
        else
            r=m;
    }
    
    cint y=y_left+r;
    cint x=x_left+y_pos-r;
    const bool flag_pos=(y<y_right && (x>=x_right || local_xs[y]<local_xs[x]));
    res[global_id]=local_xs[flag_pos?y:x];
}

__kernel void merge_init(__global const float* xs,
                         __global int* bi_y_l,
                         __global int* bi_y_r,
                         __global int* bi_x_l,
                         __global int* bi_x_r,
                         cint n, cint step)
{
    cint global_id=get_global_id(0);
    cint wg_count=n/WG_SIZE;
    if(global_id>=wg_count)return;
    
    if(global_id==0)
    {
        bi_y_l[0]=0;
        bi_x_l[0]=step;
        bi_y_r[wg_count-1]=n-step;
        bi_x_r[wg_count-1]=n;
        return;
    }
    
    cint block_step=2*step/WG_SIZE;
    if(global_id%block_step==0)
    {
        cint block_id=global_id/block_step;
        cint x=block_id*2*step;
        bi_y_l[global_id]=x;
        bi_x_l[global_id]=x+step;
        bi_y_r[global_id-1]=x-step;
        bi_x_r[global_id-1]=x;
        return;
    }
    
    cint y_left=(global_id*WG_SIZE)/(2*step)*(2*step);
    cint y_right=y_left+step;
    cint x_left=y_right;
    cint x_right=x_left+step;
    
    cint y_pos=(global_id*WG_SIZE)-y_left;
    cint delta=y_pos-step;
    cint offset=x_left+y_pos-1;
    int l=max(delta,0)-1;
    int r=min(y_pos,step);
    int m;
    while(l+1<r)
    {
        m=(l+r)/2;
        if(xs[y_left+m]<xs[offset-m])
            l=m;
        else
            r=m;
    }
    
    cint y=min(y_left+r,y_right);
    cint x=min(x_left+y_pos-r,x_right);
    bi_y_l[global_id]=y;
    bi_x_l[global_id]=x;
    bi_y_r[global_id-1]=y;
    bi_x_r[global_id-1]=x;
}

__kernel void merge_global(__global const float* xs,
                           __global float* res,
                           __global const int* bi_y_l,
                           __global const int* bi_y_r,
                           __global const int* bi_x_l,
                           __global const int* bi_x_r,
                           cint n, cint step)
{
    cint global_id=get_global_id(0);
    cint group_id=get_group_id(0);
    cint local_id=get_local_id(0);
    
    cint y_left=bi_y_l[group_id];
    cint y_right=bi_y_r[group_id];
    cint x_left=bi_x_l[group_id];
    cint x_right=bi_x_r[group_id];
    
    cint y_step=y_right-y_left;
    cint x_step=x_right-x_left;
    
    __local float local_xs[WG_SIZE];
    local_xs[local_id]=xs[(local_id<y_step)?(y_left+local_id):(x_left+local_id-y_step)];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    cint delta=(int)local_id-x_step;
    cint offset=y_step+local_id-1;
    int l=max(delta,0)-1;
    int r=min(local_id,y_step);
    int m;
    while(l+1<r)
    {
        m=(l+r)/2;
        if(local_xs[m]<local_xs[offset-m])
            l=m;
        else
            r=m;
    }
    
    cint y=r;
    cint x=y_step+local_id-r;
    const bool flag_pos=(y<y_step && (x>=WG_SIZE || local_xs[y]<local_xs[x]));
    res[global_id]=local_xs[flag_pos?y:x];
}
