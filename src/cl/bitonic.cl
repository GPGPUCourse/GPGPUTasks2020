#define WorkGroupSize 128

#define swapper_internal                                                       \
  unsigned int i1 = (id / s) * 2 * s + (id % s);                               \
  unsigned int i2 = i1 + s;                                                    \
                                                                               \
  if (offset + i1 < n && offset + i2 < n) {                                    \
                                                                               \
    float a = as[i1];                                                          \
    float b = as[i2];                                                          \
                                                                               \
    if (((a > b) && up) || ((a < b) && !up)) {                                 \
      as[i1] = b;                                                              \
      as[i2] = a;                                                              \
    }                                                                          \
  }

// Есть ли способо посимпатичнее это сделать ?
void swapper_local(__local float *as, unsigned int n, unsigned int offset,
                   unsigned int id, unsigned int s, bool up) {
  swapper_internal
}

void swapper_global(__global float *as, unsigned int n, unsigned int offset,
                    unsigned int id, unsigned int s, bool up){swapper_internal}

__kernel void bitonic_local(__global float *as, unsigned int n,
                            unsigned int kSize, unsigned int s_) {
  unsigned int g_id = get_global_id(0);
  unsigned int l_id = get_local_id(0);

  bool up = (g_id / kSize) % 2 == 0;

  __local float as_local[2 * WorkGroupSize];

  if (2 * g_id < n)
    as_local[2 * l_id] = as[2 * g_id];
  if (2 * g_id + 1 < n)
    as_local[2 * l_id + 1] = as[2 * g_id + 1];

  barrier(CLK_LOCAL_MEM_FENCE);

  unsigned int offset = (g_id / get_local_size(0)) * get_local_size(0);

  for (unsigned int s = s_; s > 0; s >>= 1) {
    swapper_local(as_local, n, offset, l_id, s, up);

    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (2 * g_id < n)
    as[2 * g_id] = as_local[2 * l_id];
  if (2 * g_id + 1 < n)
    as[2 * g_id + 1] = as_local[2 * l_id + 1];
}

// Это не будет работать, если длина массива на 2^n
// пытался придумать как решить, самое простое это видимо дополнить массив
// т.к. думать о том как надо менять местами индексы и т.д. уж что-то не вышло
__kernel void bitonic(__global float *as, unsigned int n, unsigned int kSize,
                      unsigned int s) {
  unsigned int g_id = get_global_id(0);

  bool up = (g_id / kSize) % 2 == 0;

  swapper_global(as, n, 0, g_id, s, up);
}
