#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void sum( const unsigned int BlockSize,
                   __global const unsigned int *BufferSrc,
                   __global unsigned int *BufferRes,
                   unsigned int n )
{
  unsigned int Res = 0;

  for (unsigned int i = 0; i < BlockSize; i++)
  {
    const unsigned int Index = get_group_id(0) * get_local_size(0) * BlockSize +
                               i * get_local_size(0) + get_local_id(0);

    if (Index >= n)
      break;

    Res += BufferSrc[Index];
  }

  BufferRes[get_global_id(0)] = Res;
}

// TODO
