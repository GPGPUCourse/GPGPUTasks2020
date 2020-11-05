#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

bool bitAt(unsigned int x, unsigned int bit) {
    return (x & (1 << bit)) != 0;
}

kernel void radix(global unsigned int *S, global unsigned int *A1, global unsigned int *A2, unsigned int log2n, unsigned int bit) {
    int gid = get_global_id(0);
    
    unsigned int x = A1[gid];
    bool isOne = bitAt(x, bit);
    unsigned int sum = 0;
    unsigned int o = 0;
    for (int i = 0; i <= log2n; i++) {
        unsigned int b = bitAt(gid, log2n - i);
        
        sum += S[(1 << i) - 1 + o] * b;
        o = (o + b) * 2;
    }
    
    unsigned int allOnes = S[0];
    unsigned int pos = ((1 << log2n) - allOnes + sum) * isOne + (gid - sum) * (!isOne);
    
    A2[pos] = x;
}

void tree_bit_sum(global unsigned int *S, global const unsigned int *A1, int log2n, int level, unsigned int bit) {
    int gid = get_global_id(0);
    int tid = (1 << level) - 1 + gid;
    int c1 = 2*tid + 1;
    int c2 = 2*tid + 2;
    
    if (level == log2n)
        S[tid] = bitAt(A1[gid], bit);
    else
        S[tid] = S[c1] + S[c2];
}

kernel void bit_sums(global unsigned int *S, global const unsigned int *A1, unsigned int log2n, unsigned int level, unsigned int bit) {
    if ((1 << level) == WORK_GROUP_SIZE) {
        for (int l = level; l >= 0; l--) {
            if (get_global_id(0) < (1 << l))
                tree_bit_sum(S, A1, log2n, l, bit);
            barrier(CLK_GLOBAL_MEM_FENCE);
        }
    }
    else {
        tree_bit_sum(S, A1, log2n, level, bit);
    }
}