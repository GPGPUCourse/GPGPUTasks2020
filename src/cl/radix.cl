// BLOCK_SIZE_CD >= POW2_VALUE

__kernel void CountDigits( __global const unsigned int* Array, __global unsigned int* Dst,
                           unsigned int n, unsigned int DstSize, unsigned int Power2 )
{
  __local unsigned int LocalArray[GROUP_SIZE * BLOCK_SIZE_CD];

  for (unsigned int i = 0; i < BLOCK_SIZE_CD; i++)
  {
    const unsigned int Index = get_group_id(0) * GROUP_SIZE * BLOCK_SIZE_CD +
                               i * GROUP_SIZE + get_local_id(0);

    if (Index >= n)
      break;

    LocalArray[i * GROUP_SIZE + get_local_id(0)] = Array[Index];
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  unsigned int ResCount[POW2_VALUE] = {};

  for (unsigned int i = 0; i < BLOCK_SIZE_CD; i++)
  {
    const unsigned int Index = BLOCK_SIZE_CD * get_local_id(0) + i;

    if (get_group_id(0) * GROUP_SIZE * BLOCK_SIZE_CD + Index >= n)
      break;

    ResCount[(LocalArray[Index] >> Power2) & MASK]++;
  }

  if (get_global_id(0) >= DstSize)
    return;

  for (unsigned int i = 0; i < POW2_VALUE; i++)
    Dst[DstSize * i + get_global_id(0)] = ResCount[i];
}

__kernel void EvalOffsets( __global const unsigned int* Array, unsigned int OffsetA,
                           __global unsigned int* Dst, unsigned int OffsetD,
                           unsigned int DstSize, unsigned int n )
{
  __local unsigned int LocalArray[GROUP_SIZE * BLOCK_SIZE_EO * POW2_VALUE];

  for (unsigned int i = 0; i < POW2_VALUE; i++)
    for (unsigned int j = 0; j < BLOCK_SIZE_EO; j++)
    {
      const unsigned int Index = i * n + get_group_id(0) * GROUP_SIZE * BLOCK_SIZE_EO +
                                 j * GROUP_SIZE + get_local_id(0);

      if (get_group_id(0) * GROUP_SIZE * BLOCK_SIZE_EO + j * GROUP_SIZE + get_local_id(0) >= n)
        break;

      LocalArray[i * GROUP_SIZE * BLOCK_SIZE_EO + j * GROUP_SIZE + get_local_id(0)] = Array[OffsetA + Index];
    }

  barrier(CLK_LOCAL_MEM_FENCE);

  unsigned int ResOffset[POW2_VALUE];

  for (unsigned int i = 0; i < POW2_VALUE; i++)
  {
    ResOffset[i] = 0;

    for (unsigned int j = 0; j < BLOCK_SIZE_EO; j++)
    {
      if (get_group_id(0) * GROUP_SIZE * BLOCK_SIZE_EO + j + get_local_id(0) * BLOCK_SIZE_EO >= n)
        break;

      ResOffset[i] += LocalArray[i * GROUP_SIZE * BLOCK_SIZE_EO + j + get_local_id(0) * BLOCK_SIZE_EO];
    }
  }

  if (get_global_id(0) < DstSize)
    for (unsigned int i = 0; i < POW2_VALUE; i++)
      Dst[OffsetD + DstSize * i + get_global_id(0)] = ResOffset[i];
}

// BLOCK_SIZE_EO = 2

__kernel void Radix( __global const unsigned int* Array, __global unsigned int* Dst,
                     __global const unsigned int *Offsets,
                     __global const unsigned int *Sums, unsigned int n, unsigned int Power2,
                     unsigned int SumsInd )
{
  unsigned int IndForOffsets = get_global_id(0);
  unsigned int GlobalOffsets[POW2_VALUE] = {};
  unsigned int OffsetIndex = 0;

  if (get_global_id(0) * BLOCK_SIZE_CD < n)
  {
    while (IndForOffsets > 0)
    {
      unsigned int IndDigit = (IndForOffsets & 1);

      if (IndDigit)
        for (unsigned int i = 0; i < POW2_VALUE; i++)
          GlobalOffsets[i] += Sums[Offsets[OffsetIndex * POW2_VALUE + i] + IndForOffsets - 1];

      IndForOffsets >>= 1;
      OffsetIndex++;
    }

    unsigned int GlobalSum = 0;

    for (unsigned int i = 1; i < POW2_VALUE; i++)
    {
      GlobalSum += Sums[SumsInd + i - 1];
      GlobalOffsets[i] += GlobalSum;
    }
  }

  __local unsigned int LocalArray[GROUP_SIZE * BLOCK_SIZE_CD];

  for (unsigned int i = 0; i < BLOCK_SIZE_CD; i++)
  {
    const unsigned int Index = get_group_id(0) * GROUP_SIZE * BLOCK_SIZE_CD +
                               i * GROUP_SIZE + get_local_id(0);

    if (Index >= n)
      break;

    LocalArray[i * GROUP_SIZE + get_local_id(0)] = Array[Index];
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  for (unsigned int i = 0; i < BLOCK_SIZE_CD; i++)
  {
    const unsigned int Index = BLOCK_SIZE_CD * get_local_id(0) + i;

    if (get_group_id(0) * GROUP_SIZE * BLOCK_SIZE_CD + Index >= n)
      break;

    unsigned int Elem = LocalArray[Index];

    Dst[GlobalOffsets[(Elem >> Power2) & MASK]++] = Elem;
  }
}
