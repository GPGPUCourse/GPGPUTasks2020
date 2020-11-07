#define INF 10000000

__kernel void copy(__global float *b, __global float *a) {
  unsigned int id = get_global_id(0);
  b[id] = a[id];
}

/** just for sefety */
float get_val(__global float *input, int size, int a, int x) {
  if (x >= size)
    return INF;
  if (x < 0)
    return -INF;
  return input[a + x];
}

__kernel void merge(__global float *input, __global float *output, int size) {
  /** column array start */
  int a = (get_global_id(0) / 2 / size) * 2 * size;
  /** row array start */
  int b = a + size;
  /** diagonal id */
  int d_id = get_global_id(0) % (2 * size);

  /** diagonal indices */
  int start = 0;
  int stop = d_id + 1;

  /** trying to bound our indices inside a-b square */
  if (d_id >= size) {
    start = d_id - size;
    stop = stop - start;
  }

  int middle = (start + stop) / 2;

  while (stop != start) {

    /** from our diagonal indexing A: x -1, B: d_id - x */
    bool val = get_val(input, size, a, middle - 1) <=
               get_val(input, size, b, d_id - middle);

    /** true == 1, false = 0 in our lectures */
    start = val ? middle : start;
    stop = val ? stop : middle;

    middle = (start + stop) / 2;
    if (start == stop - 1)
      break;
  }

  /* checking which array index should we take
   * for array A we need: ans -1 + 1 because our value is on the right of our
   * point */
  if (get_val(input, size, a, start) <= get_val(input, size, b, d_id - start))
    output[get_global_id(0)] = input[a + start];
  else
    output[get_global_id(0)] = input[b + d_id - start];
}
