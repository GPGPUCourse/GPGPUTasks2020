#define TILE 16

__kernel void matrix_transpose(__global const float *idata,
                               __global float *odata, unsigned int H, const unsigned int W) {

  int x = get_global_id(0);
  int y = get_global_id(1);

  if(y*W + x >= W*H)
    return;

  int local_x = get_local_id(0);
  int local_y = get_local_id(1);
 
  // LOL если тут забыть указать тип то он по дефолту или int или unit, забыл это сделать, потратил кучу времени на поиск ошибки (возможно это только в моём случае так, но это мега странно) (ещё одно причина по которой 99% людей выбирают cuda :D )
  __local float tile[TILE+1][TILE+1];
  
  tile[local_y][local_x] = idata[y * W +x];
  
  barrier(CLK_LOCAL_MEM_FENCE);
  
  int out_x = get_group_id(0) * TILE + local_y;
  int out_y = get_group_id(1) * TILE + local_x;

  odata[out_x*H + out_y] =  tile[local_x][local_y];
}
