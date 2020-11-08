#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

kernel void merge_local(global float *A) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    
    local float lA[2][WORK_GROUP_SIZE];
    
    lA[0][lid] = A[gid];
    
    int cur = 0;
    int next = 1;
    for (int k = 2; k <= WORK_GROUP_SIZE; k *= 2) {
        int b1 = lid / k * k;
        int b2 = b1 + k / 2;
    
        bool left = lid < b2;
    
        int nb = left ? b2 : b1;
        int l = nb - 1;
        int r = left ? b2 + k / 2 : b2;
        float x = lA[cur][lid];
    
        while (r - l > 1) {
            int m = (l + r) / 2;
            float y = lA[cur][m];
            if (y > x || (!left && y >= x))
                r = m;
        
            else
                l = m;
        }
    
        int pos = b1 + (lid - b1) * left + (lid - b2) * !left + r - nb;
        //printf("lid=%d b1=%d b2=%d l=%d r=%d h=%d pos=%d x=%f\n", lid, b1, b2, l, r, r - nb, pos, x);
    
        lA[next][pos] = x;
    
        cur ^= 1;
        next ^= 1;
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    A[gid] = lA[cur][lid];
}

kernel void merge(global float *A, global float *B, unsigned int n, int k) {
    int gid = get_global_id(0);
    
    int b1 = gid / k * k;
    int b2 = b1 + k / 2;
    
    bool left = gid < b2;
    
    int nb = left ? b2 : b1;
    int l = nb - 1;
    int r = left ? b2 + k / 2 : b2;
    float x = A[gid];
    
    while (r - l > 1) {
        int m = (l + r) / 2;
        float y = A[m];
        if (y > x || (!left && y >= x))
            r = m;
        
        else
            l = m;
    }
    
    int pos = b1 + (gid - b1) * left + (gid - b2) * !left + r - nb;
    //printf("gid=%d b1=%d b2=%d l=%d r=%d h=%d pos=%d\n", gid, b1, b2, l, r, r - nb, pos);
    
    B[pos] = x;
}