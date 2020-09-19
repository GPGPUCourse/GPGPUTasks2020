// lines were deleted because CLion is an extremly unusable text editor
#line 3 // Седьмая строчка теперь восьмая (при ошибках компиляции в логе
        // компиляции будут указаны корректные строчки благодаря этой директиве)

__kernel void aplusb(__global const float *a, __global const float *b,
                     __global float *c, unsigned int n) {
  unsigned int id = get_global_id(0);

  if (id > n)
    return;
  c[id] = a[id] + b[id];
}
