#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6


/*
__kernel void max_prefix_sum_tree(__global int *inout_sum, __global int *inout_prefix,  unsigned int n)
{
    __local int max_prefix[2][WORK_GROUP_SIZE];
    __local int sum[2][WORK_GROUP_SIZE];

    int id = get_global_id(0);
    int local_id = get_local_id(0);
    
    sum[0][local_id] = inout_sum[id];
    max_prefix[0][local_id] = inout_prefix[id];

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = WORK_GROUP_SIZE / 2; i >= 1; i /= 2) {
        if (local_id < i) {
            int id1 = local_id * 2;
            int id2 = local_id * 2 + 1;
    
            int s1 = sumA[id1];
            int s2 = sumA[id2];
    
            sumB[local_id] = s1 + s2;
            max_prefixB[local_id] = max(max_prefixA[id1], s1 + max_prefixA[id2]);
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (local_id == 0) {
        int out_id = id / WORK_GROUP_SIZE;
        inout_sum[out_id] = sumA[0];
        inout_prefix[out_id] = max_prefixA[0];
    }
}*/


__kernel void max_prefix_sum(__global int *in_sum, __global int *in_prefix, __global int *in_pid,
                             __global int *out_sum, __global int *out_prefix, __global int *out_pid,
                             unsigned int n)
{
    __local int max_prefix[WORK_GROUP_SIZE];
    __local int sum[WORK_GROUP_SIZE];
    __local int pid[WORK_GROUP_SIZE];
    
    int id = get_global_id(0);
    int local_id = get_local_id(0);
    
    sum[local_id] = in_sum[id];
    max_prefix[local_id] = in_prefix[id];
    pid[local_id] = in_pid[id];
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (local_id == 0) {
        int mySum = 0;
        int myPref = 0;
        int myPid = 0;
        
        for (int i = 0; i < min((int) WORK_GROUP_SIZE, (int) (n - id)); i++) {
            if (myPref < mySum + max_prefix[i]) {
                myPref = mySum + max_prefix[i];
                myPid = pid[i];
            }
            
            //myPref = max((int) (myPref), (int) (mySum + max_prefix[i]));
            mySum += sum[i];
        }
        
        int out_id = id / WORK_GROUP_SIZE;
        
        out_sum[out_id] = mySum;
        out_prefix[out_id] = myPref;
        out_pid[out_id] = myPid;
    }
}