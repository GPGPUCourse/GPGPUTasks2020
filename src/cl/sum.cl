#line 2

#define WORK_SIZE 128

__kernel void sum(__global const unsigned int *a, unsigned int n,
                  __global unsigned int *r) {

  int l_i = get_local_id(0);

  __local unsigned int local_a[WORK_SIZE];

  if (get_global_id(0) >= n)
    local_a[l_i] = 0;
  else
    local_a[l_i] = a[get_global_id(0)];

  barrier(CLK_LOCAL_MEM_FENCE);

  for (int step = WORK_SIZE; step > 1; step /= 2) {
    if (2 * l_i < step) {
      local_a[l_i] += local_a[l_i + step / 2];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (l_i == 0)
    atomic_add(r, local_a[0]);
}
