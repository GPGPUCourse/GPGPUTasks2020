// BLOCK_SIZE == get_local_size(0), BLOCK_SIZE == get_local_size(1)
#define BLOCK_SIZE 8

#define DEBUG 0

__kernel void matrix_transpose( __global const float *Matr, __global float *Dest, unsigned int H, unsigned int W )
{
  __local float Block[BLOCK_SIZE][BLOCK_SIZE + 1];

  if (get_global_id(0) < W && get_global_id(1) < H)
  {
#if DEBUG
    printf("(%2d, %2d)\tMatr[%2d]\t->\tBlock[%2d][%2d]\tValue=%f\n", (int)get_global_id(1), (int)get_global_id(0), (int)(get_global_id(1) * W + get_global_id(0)), (int)get_local_id(0), (int)get_local_id(1), Matr[get_global_id(1) * W + get_global_id(0)]);
#endif
    Block[get_local_id(0)][get_local_id(1)] = Matr[get_global_id(1) * W + get_global_id(0)];
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  if (get_group_id(0) * BLOCK_SIZE + get_local_id(1) < W && get_group_id(1) * BLOCK_SIZE + get_local_id(0) < H)
  {
#if DEBUG
    printf("(%2d, %2d)\tBlock[%2d][%2d]\t->\tDest[%2d]\tValue=%f\n", (int)get_global_id(1), (int)get_global_id(0), (int)get_local_id(1), (int)get_local_id(0), (int)(H * (get_group_id(0) * BLOCK_SIZE + get_local_id(1)) + get_group_id(1) * BLOCK_SIZE + get_local_id(0)), Block[get_local_id(1)][get_local_id(0)]);
#endif
    Dest[H * (get_group_id(0) * BLOCK_SIZE + get_local_id(1)) + get_group_id(1) * BLOCK_SIZE + get_local_id(0)] =
      Block[get_local_id(1)][get_local_id(0)];
  }
}
