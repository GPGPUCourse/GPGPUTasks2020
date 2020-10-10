#define TILE 16

__kernel void matrix_multiplication(__global const float *Adata,
                                    __global const float *Bdata,
                                    __global float *res, unsigned int HA,
                                    const unsigned int W,
                                    const unsigned int WB) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  int local_x = get_local_id(0);
  int local_y = get_local_id(1);

  __local float tileA[TILE][TILE + 1], tileB[TILE][TILE + 1];

  __local float sum[TILE][TILE + 1];
  sum[local_y][local_x] = 0;

  for (int i = 0; i * TILE < W; ++i) {
    tileA[local_y][local_x] = Adata[y * W + (i * TILE + local_x)];
    tileB[local_y][local_x] = Bdata[x + (i * TILE + local_y) * WB];

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k = 0; k < TILE; ++k) {
      sum[local_y][local_x] += tileA[local_y][k] * tileB[k][local_x];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

  }

  res[y * WB + x] = sum[local_y][local_x];
}
