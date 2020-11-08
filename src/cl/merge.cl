int Check( __global const float *Src1, __global const float *Src2,
            int Size1, int Size2,
            int x1, int x2 )
{
  if (x1 < 0 || x2 >= Size2)
    return 1;

  return Src1[x1] < Src2[x2];
}

void MergeLocal( __global const float *Src1, __global const float *Src2,
                 __global float *Dst, int Size1, int Size2,
                 int CurrentWork )
{
  int x1, x2, m, l, r;

  if (CurrentWork >= Size1)
  {
    x1 = Size1 - 1;
    x2 = CurrentWork - x1;
    r = Size2 - x2;
  }
  else
  {
    x1 = CurrentWork;
    x2 = 0;
    r = x1 + 1;
  }

  l = -1;
  

  while (l + 1 < r)
  {
    int m = (l + r) / 2;
    int mx1 = x1 - m;
    int mx2 = x2 + m;

    if (Check(Src1, Src2, Size1, Size2, mx1, mx2))
      r = m;
    else
      l = m;
  }

  x1 = x1 - r;
  x2 = x2 + r;

  //float Res;
  //
  //if (x2 > 0 && Check(Src1, Src2, Size1, Size2, x1, x2 - 1))
  //  Res = Src2[x2 - 1];
  //else
  //  Res = Src1[x1];
  //
  //Dst[CurrentWork] = Res;

  if (x2 > 0 && Check(Src1, Src2, Size1, Size2, x1, x2 - 1))
    Dst[CurrentWork] = Src2[x2 - 1];
  else
    Dst[CurrentWork] = Src1[x1];
}

__kernel void Merge( __global const float *Src, __global float *Dst, int n, int CurrentSize )
{
  if (get_global_id(0) >= n)
    return;

  int CurrentMerge = get_global_id(0) / CurrentSize / 2;
  int CurrentWork = get_global_id(0) % (CurrentSize * 2);

  int LocalSize1 = max((int)0, (int)min((int)CurrentSize,
    (int)(n - (CurrentMerge * 2 * CurrentSize))));
  int LocalSize2 = max((int)0, (int)min((int)CurrentSize,
    (int)(n - (CurrentMerge * 2 * CurrentSize + CurrentSize))));

  __global const float *LocalDst = Dst + CurrentMerge * 2 * CurrentSize;
  __global const float *LocalSrc1 = Src + CurrentMerge * 2 * CurrentSize;
  __global const float *LocalSrc2 = Src + CurrentMerge * 2 * CurrentSize + LocalSize1;

  MergeLocal(LocalSrc1, LocalSrc2, LocalDst, LocalSize1, LocalSize2, CurrentWork);
}
