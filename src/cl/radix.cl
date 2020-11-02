#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif



kernel void sums(global unsigned int *S, unsigned int t, unsigned int level) {
    int id = get_global_id(0);
    
    if ((1 << level) * id < (1 << t)) {
        int old_id = (1 << (t + 1)) - 2 - (1 << (t - level + 1)) + id;
        
        unsigned int sum = S[2 * old_id] + S[2*old_id + 1];
        
        int new_id = (1 << (t + 1)) - 2 - (1 << (t - level)) + id;
        
        S[new_id] = sum;
    }
}

void local_prefix_sums(local unsigned int *lA, local unsigned int *lOut, local unsigned int *out_sum) {
    int sum = 0;
    for (int i = 0; i < WORK_GROUP_SIZE; i++) {
        lOut[i] = sum;
        sum += lA[i];
    }
    *out_sum = sum;
}

kernel void local_radix(global unsigned int* as, global unsigned int *out, unsigned int n)
{
    int id = get_global_id(0);
    int lid = get_local_id(0);
    
    local unsigned int lA[2][WORK_GROUP_SIZE];
    local unsigned int zeros[WORK_GROUP_SIZE];
    local unsigned int ones[WORK_GROUP_SIZE];
    
    local unsigned int prefOnes[WORK_GROUP_SIZE];
    local unsigned int prefZeros[WORK_GROUP_SIZE];
    
    local unsigned int t1;
    local unsigned int t2;
    
    lA[0][lid] = as[id];
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    const int bit_count = 32;
    for (int i = 0; i < bit_count; i++) {
        int cur = i & 1;
        int next = (i + 1) & 1;
        
        if ((lA[cur][lid] & (1 << i)) == 0) {
            zeros[lid] = 1;
            ones[lid] = 0;
        }
        else {
            ones[lid] = 1;
            zeros[lid] = 0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    
        if (lid == WORK_GROUP_SIZE - 1) {
            local_prefix_sums(ones, prefOnes, &t1);
            local_prefix_sums(zeros, prefZeros, &t2);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        if (ones[lid]) {
            lA[next][t2 + prefOnes[lid]] = lA[cur][lid];
        }
        else {
            lA[next][prefZeros[lid]] = lA[cur][lid];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    out[id] = lA[bit_count & 1][lid];
}
