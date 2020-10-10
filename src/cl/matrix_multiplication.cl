#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

kernel void matrix_multiplication(global const float *A, global const float *B, global float *C,
                                  unsigned int M, unsigned int K, unsigned int N)
{
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    int lidx = get_local_id(0);
    int lidy = get_local_id(1);
    
    local float lA[WORK_GROUP_SIDE][WORK_GROUP_SIDE];
    local float lB[WORK_GROUP_SIDE][WORK_GROUP_SIDE];
    local float lC[WORK_GROUP_SIDE][WORK_GROUP_SIDE];
    
    lC[lidy][lidx] = 0;
    // No barrier needed
    
    //printf("KG: %d\n", K / WORK_GROUP_SIDE);
    
    for (int kg = 0; kg < K / WORK_GROUP_SIDE; kg++) {
        int aid = idy * K + (kg * WORK_GROUP_SIDE + lidx);
        int bid = (kg * WORK_GROUP_SIDE + lidy) * N + idx;
        
        lA[lidy][lidx] = A[aid];
        lB[lidy][lidx] = B[bid];
        
        //printf("idy=%d idx=%d ly=%d lx=%d Aid=%d Bid=%d\n", idy, idx, lidy, lidx, aid, bid);
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (int i = 0; i < WORK_GROUP_SIDE; i++) {
            lC[lidy][lidx] += lA[lidy][i] * lB[i][lidx];
        }
        //lC[lidy][lidx] = lB[lidy][lidx];

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    C[idy * N + idx] = lC[lidy][lidx];
}