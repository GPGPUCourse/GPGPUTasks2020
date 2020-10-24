void Sort2Local( __local float *LocalArray, unsigned int i, unsigned int j, unsigned int n ) // j > i
{
  if (i >= n || j >= n)
    return;

  float ValI = LocalArray[i];
  float ValJ = LocalArray[j];

  if (ValI > ValJ)
  {
    LocalArray[i] = ValJ;
    LocalArray[j] = ValI;
  }
}

void LocalFirstIteration( __local float *LocalArray, unsigned int HalfBlockSize, unsigned int SortSize, unsigned int n )
{
  unsigned int BlockSize = HalfBlockSize * 2;

  for (unsigned int j = get_local_id(0); j < SortSize; j += get_local_size(0))
  {
    unsigned int BlockId = j / HalfBlockSize;
    unsigned int StepId = j % HalfBlockSize;

    Sort2Local(LocalArray, BlockId * BlockSize + StepId, (BlockId + 1) * BlockSize - StepId - 1, n);
  }
}

void LocalIteration( __local float *LocalArray, unsigned int HalfBlockSize, unsigned int SortSize, unsigned int n )
{
  unsigned int BlockSize = HalfBlockSize * 2;

  for (unsigned int j = get_local_id(0); j < SortSize; j += get_local_size(0))
  {
    unsigned int BlockId = j / HalfBlockSize;
    unsigned int StepId = j % HalfBlockSize;

    Sort2Local(LocalArray, BlockId * BlockSize + StepId, BlockId * BlockSize + StepId + HalfBlockSize, n);
  }
}

__kernel void BitonicLocalFirst( __global float *Array, unsigned int n )
{
  __local float LocalArray[POW2_LOCAL_SORT_VALUE];

  unsigned int LocalN = min((unsigned int)(n - get_group_id(0) * POW2_LOCAL_SORT_VALUE), (unsigned int)POW2_LOCAL_SORT_VALUE);
  unsigned int SortSize = POW2_LOCAL_SORT_VALUE / 2;

  for (unsigned int i = get_local_id(0); i < LocalN; i += get_local_size(0))
    LocalArray[i] = Array[i + get_group_id(0) * POW2_LOCAL_SORT_VALUE];

  barrier(CLK_LOCAL_MEM_FENCE);

  for (unsigned int FirstStepSizeI = 0; FirstStepSizeI < POW2_LOCAL_SORT; FirstStepSizeI++)
  {
    LocalFirstIteration(LocalArray, 1 << FirstStepSizeI, SortSize, LocalN);
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int StepSizeI = FirstStepSizeI - 1; StepSizeI >= 0; StepSizeI--)
    {
      LocalIteration(LocalArray, 1 << StepSizeI, SortSize, LocalN);
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }

  for (unsigned int i = get_local_id(0); i < LocalN; i += get_local_size(0))
    Array[i + get_group_id(0) * POW2_LOCAL_SORT_VALUE] = LocalArray[i];
}

__kernel void BitonicLocal( __global float *Array, unsigned int n )
{
  __local float LocalArray[POW2_LOCAL_SORT_VALUE];

  unsigned int LocalN = min((unsigned int)(n - get_group_id(0) * POW2_LOCAL_SORT_VALUE), (unsigned int)POW2_LOCAL_SORT_VALUE);
  unsigned int SortSize = POW2_LOCAL_SORT_VALUE / 2;

  for (unsigned int i = get_local_id(0); i < LocalN; i += get_local_size(0))
    LocalArray[i] = Array[i + get_group_id(0) * POW2_LOCAL_SORT_VALUE];

  barrier(CLK_LOCAL_MEM_FENCE);

  for (int StepSizeI = POW2_LOCAL_SORT - 1; StepSizeI >= 0; StepSizeI--)
  {
    LocalIteration(LocalArray, 1 << StepSizeI, SortSize, LocalN);
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  for (unsigned int i = get_local_id(0); i < LocalN; i += get_local_size(0))
    Array[i + get_group_id(0) * POW2_LOCAL_SORT_VALUE] = LocalArray[i];
}

void Sort2( __global float *Array, unsigned int i, unsigned int j, unsigned int n ) // j > i
{
  if (i >= n || j >= n)
    return;

  float ValI = Array[i];
  float ValJ = Array[j];

  if (ValI > ValJ)
  {
    Array[i] = ValJ;
    Array[j] = ValI;
  }
}

__kernel void BitonicFirst( __global float *Array, unsigned int HalfBlockSize, unsigned int n )
{
  unsigned int BlockSize = HalfBlockSize * 2;
  unsigned int BlockId = get_global_id(0) / HalfBlockSize;
  unsigned int StepId = get_global_id(0) % HalfBlockSize;

  Sort2(Array, BlockId * BlockSize + StepId, (BlockId + 1) * BlockSize - StepId - 1, n);
}

__kernel void Bitonic( __global float *Array, unsigned int HalfBlockSize, unsigned int n )
{
  unsigned int BlockSize = HalfBlockSize * 2;
  unsigned int BlockId = get_global_id(0) / HalfBlockSize;
  unsigned int StepId = get_global_id(0) % HalfBlockSize;

  Sort2(Array, BlockId * BlockSize + StepId, BlockId * BlockSize + StepId + HalfBlockSize, n);
}
