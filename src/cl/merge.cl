#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define uint unsigned int
#define cuint const uint
#define WG_SIZE 256

__kernel void merge(__global const float* xs,
                    __global float* res,
                    cuint n, cuint step)
{
    cuint global_id=get_global_id(0);
    cuint y_left=global_id/(2*step)*(2*step);
    cuint y_right=y_left+step;
    cuint x_left=y_right;
    cuint x_right=x_left+step;
    
    cuint y_pos=global_id-y_left;
    const bool flag=(y_pos<step);
    
    int left=flag?-1:y_pos-step-1;
    int right=flag?y_pos:step;
    int mid;
    while(left+1<right)
    {
        mid=(left+right)/2;
        if(xs[y_left+mid]<=xs[x_left+y_pos-mid-1])
            left=mid;
        else
            right=mid;
    }
    
    cuint y=y_left+right;
    cuint x=x_left+y_pos-right;
    const bool flag_pos=(y<y_right && (x>=x_right || xs[y]<=xs[x]));
    res[global_id]=xs[flag_pos?y:x];
}

/*---------------------*/

/*__kernel void merge_local(__global const float* xs,
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
    const bool flag=(y_pos<step);
    
    int left=flag?-1:y_pos-step-1;
    int right=flag?y_pos:step;
    int mid;
    while(left+1<right)
    {
        mid=(left+right)/2;
        if(local_xs[y_left+mid]<=local_xs[x_left+y_pos-mid-1])
            left=mid;
        else
            right=mid;
    }
    
    cuint y=y_left+right;
    cuint x=x_left+y_pos-right;
    const bool flag_pos=(y<y_right && (x>=x_right || local_xs[y]<=local_xs[x]));
    res[global_id]=local_xs[flag_pos?y:x];
}

__kernel void merge_init(__global const float* xs,
                          __global float* block_inds_x,
                          __global float* block_inds_y,
                          cuint n, cuint step)
{
    cuint global_id=get_global_id(0);
    if(global_id>=n)return;
    
    cuint y_left=(global_id*WG_SIZE)/(2*step)*(2*step);
    cuint y_right=y_left+step;
    cuint x_left=y_right;
    cuint x_right=x_left+step;
    
    cuint y_pos=(global_id*WG_SIZE)-y_left;
    const bool flag=(y_pos<step);
    
    int left=flag?-1:y_pos-step-1;
    int right=flag?y_pos:step;
    int mid;
    while(left+1<right)
    {
        mid=(left+right)/2;
        if(xs[y_left+mid]<=xs[x_left+y_pos-mid-1])
            left=mid;
        else
            right=mid;
    }
    
    block_inds_y[global_id]=min(y_left+right,y_right);
    block_inds_x[global_id]=min(x_left+y_pos-right,x_right);
}

__kernel void merge_global(__global const float* xs,
                          __global float* res,
                          __global float* block_inds_x,
                          __global float* block_inds_y,
                          cuint n, cuint step)
{
    cuint global_id=get_global_id(0);
    cuint group_id=get_group_id(0);
    cuint local_id=get_local_id(0);
    
    cuint y_left=block_inds_y[group_id];
    cuint x_left=block_inds_x[group_id];
    uint y_right,x_right;
    if(group_id+1<n/WG_SIZE)
    {
        bool flag_group=((group_id+1)%(step)==0);
        y_right=block_inds_y[group_id+1]-(flag_group?step:0);
        x_right=block_inds_x[group_id+1]-(flag_group?step:0);
    }
    else
    {
        y_right=n-step;
        x_right=n;
    }
    
    cuint y_step=y_right-y_left;
    cuint x_step=x_right-x_left;
    
    __local float local_xs[WG_SIZE];
    local_xs[local_id]=xs[(local_id<y_step)?(y_left+local_id):(x_left+local_id-y_step)];

    barrier(CLK_LOCAL_MEM_FENCE);
    
    int delta=(int)local_id-x_step;
    int l=max(delta,0)-1;
    int r=min(local_id,y_step);
    int m=(l+r)/2;
    while(l+1<r)
    {
        m=(l+r)/2;
        if(local_xs[m]<=local_xs[y_step+local_id-m-1])
            l=m;
        else
            r=m;
    }
    
    cuint y=r;
    cuint x=y_step+local_id-r;
    const bool flag_pos=(y<y_step && (x>=y_step+x_step || local_xs[y]<=local_xs[x]));
    res[global_id]=local_xs[flag_pos?y:x];
}*/
