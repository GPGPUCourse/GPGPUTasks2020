#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define GROUP_SIZE (64)
#define BLOCK_SIZE (64 + 1)  // +1 for solve bank conflicts

__kernel void MaxPrefixSumFirst( __global const int *Array, __global int *Sum,
                                 __global int *MaxSum, __global unsigned int *MaxSumInd,
                                 const unsigned int n )
{
  __local int LocalArray[GROUP_SIZE * BLOCK_SIZE];

  for (unsigned int i = 0; i < BLOCK_SIZE; i++)
  {
    const unsigned int Index = get_group_id(0) * GROUP_SIZE * BLOCK_SIZE +
                               i * GROUP_SIZE + get_local_id(0);

    if (Index >= n)
      break;

    LocalArray[i * GROUP_SIZE + get_local_id(0)] = Array[Index];
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  int ResSum = 0;
  int ResMaxSum = 0;
  int ResMaxSumInd = 0;

  for (unsigned int i = 0; i < BLOCK_SIZE; i++)
  {
    const unsigned int Index = BLOCK_SIZE * get_local_id(0) + i;

    if (get_group_id(0) * GROUP_SIZE * BLOCK_SIZE + Index >= n)
      break;

    ResSum += LocalArray[Index];

    if (ResSum > ResMaxSum)
    {
        ResMaxSum = ResSum;
        ResMaxSumInd = i + 1;
    }
  }

  Sum[get_global_id(0)] = ResSum;
  MaxSum[get_global_id(0)] = ResMaxSum;
  MaxSumInd[get_global_id(0)] = ResMaxSumInd;
}

__kernel void MaxPrefixSum( __global const int *SumArray,
                            __global const int *MaxSumArray,
                            __global const int *MaxSumIndArray,
                            __global int *ResSumArray,
                            __global int *ResMaxSumArray,
                            __global unsigned int *ResMaxSumIndArray,
                            const unsigned int n, const unsigned int CellSize )
{
  __local int LocalSumArray[GROUP_SIZE * BLOCK_SIZE];
  __local int LocalMaxSumArray[GROUP_SIZE * BLOCK_SIZE];
  //__local int LocalMaxSumIndArray[GROUP_SIZE * BLOCK_SIZE];

  for (unsigned int i = 0; i < BLOCK_SIZE; i++)
  {
    const unsigned int Index = get_group_id(0) * GROUP_SIZE * BLOCK_SIZE +
                               i * GROUP_SIZE + get_local_id(0);

    if (Index >= n)
      break;

    LocalSumArray[i * GROUP_SIZE + get_local_id(0)] = SumArray[Index];
    LocalMaxSumArray[i * GROUP_SIZE + get_local_id(0)] = MaxSumArray[Index];
    //LocalMaxSumIndArray[i * GROUP_SIZE + get_local_id(0)] = MaxSumIndArray[Index];
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  int ResSum = 0;
  int ResMaxSum = 0;
  unsigned int ResMaxSumIndInArray = 0;

  for (unsigned int i = 0; i < BLOCK_SIZE; i++)
  {
    const unsigned int Index = BLOCK_SIZE * get_local_id(0) + i;

    if (get_group_id(0) * GROUP_SIZE * BLOCK_SIZE + Index >= n)
      break;

    const int TryMaxSum = ResSum + LocalMaxSumArray[Index];

    if (TryMaxSum > ResMaxSum)
    {
      ResMaxSum = TryMaxSum;
      ResMaxSumIndInArray = i;
    }

    ResSum += LocalSumArray[Index];
  }

  ResSumArray[get_global_id(0)] = ResSum;
  ResMaxSumArray[get_global_id(0)] = ResMaxSum;
  ResMaxSumIndArray[get_global_id(0)] = CellSize * ResMaxSumIndInArray +
    MaxSumIndArray[get_global_id(0) * BLOCK_SIZE + ResMaxSumIndInArray];
}
