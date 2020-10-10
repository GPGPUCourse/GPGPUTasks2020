// BLOCK_SIZE == get_local_size(0), BLOCK_SIZE == get_local_size(1)
#define BLOCK_SIZE 8

#define DEBUG 0

__kernel void matrix_multiplication( __global const float *MatrA, __global const float *MatrB,
  __global float *Dest, unsigned int H1, unsigned int K, unsigned int W2 )
{
  __local float BlockA[BLOCK_SIZE][BLOCK_SIZE + 1];
  __local float BlockB[BLOCK_SIZE][BLOCK_SIZE + 1];

  float Res = 0;
  int BlockI;
  int i;

  for (BlockI = 0; BlockI < K; BlockI += BLOCK_SIZE)
  {
    if (get_local_id(0) + BlockI < K && get_global_id(1) < H1)
    {
#if DEBUG
      printf("(%2d, %2d), %d\tMatrA[%2d]\t->\tBlockA[%2d][%2d]\tValue=%5f\n", (int)get_global_id(1), (int)get_global_id(0), BlockI, (int)(get_global_id(1) * K + get_local_id(0) + BlockI), (int)get_local_id(1), (int)get_local_id(0), MatrA[get_global_id(1) * K + get_local_id(0) + BlockI]);
#endif
      BlockA[get_local_id(1)][get_local_id(0)] =
        MatrA[get_global_id(1) * K + get_local_id(0) + BlockI];
    }
    else
      BlockA[get_local_id(1)][get_local_id(0)] = 0;

    if (get_global_id(0) < W2 && BlockI + get_local_id(1) < K)
    {
#if DEBUG
      printf("(%2d, %2d), %d\tMatrB[%2d]\t->\tBlockB[%2d][%2d]\tValue=%5f\n", (int)get_global_id(1), (int)get_global_id(0), BlockI, (int)((BlockI + get_local_id(1)) * W2 + get_global_id(0)), (int)get_local_id(0), (int)get_local_id(1), MatrB[(BlockI + get_local_id(1)) * W2 + get_global_id(0)]);
#endif
      BlockB[get_local_id(0)][get_local_id(1)] =
        MatrB[(BlockI + get_local_id(1)) * W2 + get_global_id(0)];
    }
    else
      BlockB[get_local_id(0)][get_local_id(1)] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (i = 0; i < BLOCK_SIZE; i++)
    {
#if DEBUG
      printf("(%2d, %2d)\ti=%2d\tOldRes=%5f\tValue = %5f * %5f = %5f\n", (int)get_global_id(1), (int)get_global_id(0), i, Res, BlockA[get_local_id(1)][i], BlockB[get_local_id(0)][i], BlockA[get_local_id(1)][i] * BlockB[get_local_id(0)][i]);
#endif
      Res += BlockA[get_local_id(1)][i] * BlockB[get_local_id(0)][i];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (get_global_id(1) < H1 && get_global_id(0) < W2)
    Dest[get_global_id(1) * W2 + get_global_id(0)] = Res;
}

#define THREAD_WORK_SIZE 4
#define WORK_BLOCK_SIZE (BLOCK_SIZE * THREAD_WORK_SIZE)

__kernel void matrix_multiplication2( __global const float *MatrA, __global const float *MatrB,
  __global float *Dest, unsigned int H1, unsigned int K, unsigned int W2 )
{
  __local float BlockA[WORK_BLOCK_SIZE][WORK_BLOCK_SIZE + 1];
  __local float BlockB[WORK_BLOCK_SIZE][WORK_BLOCK_SIZE + 1];

  float Res[THREAD_WORK_SIZE][THREAD_WORK_SIZE];

  int WorkI;
  int WorkJ;

  for (WorkI = 0; WorkI < THREAD_WORK_SIZE; WorkI++)
    for (WorkJ = 0; WorkJ < THREAD_WORK_SIZE; WorkJ++)
      Res[WorkI][WorkJ] = 0;

  int BlockI;
  int i;

  const int BlockSize2 = BLOCK_SIZE * BLOCK_SIZE;
  const int ThreadWork2 = THREAD_WORK_SIZE * THREAD_WORK_SIZE;
  const int LocalId = get_local_id(1) * BLOCK_SIZE + get_local_id(0);

  for (BlockI = 0; BlockI < K; BlockI += WORK_BLOCK_SIZE)
  {
    for (i = 0; i < ThreadWork2; i++)
    {
      const int Cur = BlockSize2 * i + LocalId;
      const int Local0 = Cur % WORK_BLOCK_SIZE;
      const int Local1 = Cur / WORK_BLOCK_SIZE;
      const int Global0 = get_group_id(0) * WORK_BLOCK_SIZE + Local0;
      const int Global1 = get_group_id(1) * WORK_BLOCK_SIZE + Local1;

      if (Local0 + BlockI < K && Global1 < H1)
      {
#if DEBUG
        printf("A (%2d, %2d) : (%2d, %2d), (%2d, %2d), %d, Id=%02d\n", (int)get_global_id(1), (int)get_global_id(0), Global1, Global0, Local1, Local0, i, LocalId);
#endif
        BlockA[Local1][Local0] =
          MatrA[Global1 * K + Local0 + BlockI];
      }
      else
        BlockA[Local1][Local0] = 0;

      if (Global0 < W2 && BlockI + Local1 < K)
      {
#if DEBUG
        printf("B (%2d, %2d) : (%2d, %2d), (%2d, %2d), %d, Id=%02d\n", (int)get_global_id(1), (int)get_global_id(0), Global1, Global0, Local1, Local0, i, LocalId);
#endif
        BlockB[Local0][Local1] =
          MatrB[(BlockI + Local1) * W2 + Global0];
      }
      else
        BlockB[Local0][Local1] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    float TileB[THREAD_WORK_SIZE];

    for (i = 0; i < WORK_BLOCK_SIZE; i++)
    {
      for (WorkJ = 0; WorkJ < THREAD_WORK_SIZE; WorkJ++)
        TileB[WorkJ] = BlockB[get_local_id(0) * THREAD_WORK_SIZE + WorkJ][i];

      for (WorkI = 0; WorkI < THREAD_WORK_SIZE; WorkI++)
      {
        const float ValA = BlockA[get_local_id(1) * THREAD_WORK_SIZE + WorkI][i];

        for (WorkJ = 0; WorkJ < THREAD_WORK_SIZE; WorkJ++)
          Res[WorkI][WorkJ] += ValA * TileB[WorkJ];
      }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  for (WorkI = 0; WorkI < THREAD_WORK_SIZE; WorkI++)
    for (WorkJ = 0; WorkJ < THREAD_WORK_SIZE; WorkJ++)
      BlockA[get_local_id(1) * THREAD_WORK_SIZE + WorkI][get_local_id(0) * THREAD_WORK_SIZE + WorkJ] =
        Res[WorkI][WorkJ];

  barrier(CLK_LOCAL_MEM_FENCE);

  for (i = 0; i < ThreadWork2; i++)
  {
    const int Cur = BlockSize2 * i + LocalId;
    const int Local0 = Cur % WORK_BLOCK_SIZE;
    const int Local1 = Cur / WORK_BLOCK_SIZE;
    const int Global0 = get_group_id(0) * WORK_BLOCK_SIZE + Local0;
    const int Global1 = get_group_id(1) * WORK_BLOCK_SIZE + Local1;

    if (Global0 < W2 && Global1 < H1)
      Dest[Global1 * W2 + Global0] = BlockA[Local1][Local0];
  }
}
