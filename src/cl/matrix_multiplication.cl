#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 16
__kernel void matrix_multiplication(
                               __global const float* as,
                               __global const float* bs,
                               __global float* cs,
                               int M,
                               int K,
                               int N)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);
    __local float tileA[WORK_GROUP_SIZE + 1][WORK_GROUP_SIZE + 1];
    __local float tileB[WORK_GROUP_SIZE + 1][WORK_GROUP_SIZE + 1];

   float sum = 0.0f;
   for (int tileK = 0; tileK * WORK_GROUP_SIZE < K; ++tileK) {
        if (i < M && tileK * WORK_GROUP_SIZE + local_j < K) {
            tileA[local_i][local_j] = as[i * K + tileK * WORK_GROUP_SIZE + local_j];
        } else {
            tileA[local_i][local_j] = 0;
        }

        if (j < N && (tileK * WORK_GROUP_SIZE + local_i) < K) {
            tileB[local_i][local_j] = bs[(tileK * WORK_GROUP_SIZE + local_i) * N + j];
        } else {
            tileB[local_i][local_j] = 0;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < WORK_GROUP_SIZE; ++k) {
            sum += tileA[local_i][k] * tileB[k][local_j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
   }

   if (i < M && j < N) {
        cs[i * N + j] = sum;
    }
}