 
#define WORK_GROUP_SIZE 256


__kernel void bitonic(__global float* as, int n, unsigned  batch_size, unsigned size)
{

    const unsigned gx = get_global_id(0);
    bool flag = ((gx / size) % 2) == 0;

    if((gx  + batch_size < n) && (((gx / batch_size) % 2) == 0)){
        float a = as[gx];
        float b = as[gx + batch_size];
        if((a > b) == flag){
            as[gx] = b;
            as[gx + batch_size] = a;
        }
    }


}


__kernel void local_bitonic(__global float* as, int n, unsigned  batch_size, unsigned size)
{

    const unsigned gx = get_global_id(0);
    const unsigned lx = get_local_id(0);

    
    __local float localAs[WORK_GROUP_SIZE];
    localAs[lx] = as[gx];

    barrier(CLK_LOCAL_MEM_FENCE);

    bool flag = ((gx / size) % 2) == 0;


    for(;batch_size > 0;batch_size >>= 1){
        if((gx  + batch_size < n) && (((gx / batch_size) % 2) == 0)){
            float a = localAs[lx];
            float b = localAs[lx + batch_size];
            if((a > b) == flag){
                localAs[lx] = b;
                localAs[lx + batch_size] = a;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    as[gx] = localAs[lx];

}