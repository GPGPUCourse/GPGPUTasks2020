#define TILE_SIZE 16

__kernel void matrix_transpose(__global float *a, __global float *at, unsigned int m, unsigned int k)
{
    unsigned int x = get_global_id(0);
    unsigned int y = get_global_id(1);
    unsigned int xloc = get_local_id(0);
    unsigned int yloc = get_local_id(1);

    __local float buf[TILE_SIZE][TILE_SIZE];
    buf[yloc][xloc] = (x < k && y < m) ? a[y * k + x] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (xloc < yloc) {
        float tmp = buf[yloc][xloc];
        buf[yloc][xloc] = buf[xloc][yloc];
        buf[xloc][yloc] = tmp;
    }

    unsigned int yn = y + xloc - yloc;
    unsigned int xn = x + yloc - xloc;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (xn < k && yn < m) at[xn * m + yn] = buf[yloc][xloc];
}