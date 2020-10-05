#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6


#define WORK_GROUP_SIZE 256


__kernel void sum_fast(__global const unsigned int *xs,
                       __global unsigned int *sum,
                        unsigned int n){

    const unsigned local_id = get_local_id(0);
    const unsigned global_id = get_global_id(0);

    __local unsigned int localXs[WORK_GROUP_SIZE];
    if(global_id >= n){
        localXs[local_id] = 0;
    }
    else{
        localXs[local_id] = xs[global_id];
    }

    barrier(CLK_LOCAL_MEM_FENCE);;

    for(int nvalues = WORK_GROUP_SIZE; nvalues > 1;nvalues = nvalues/2){

        if(2 * local_id < nvalues){
            int left = localXs[local_id];
            int right = localXs[local_id + nvalues/2];

            localXs[local_id] = left + right;
        }
        barrier(CLK_LOCAL_MEM_FENCE);;

       
    }
     if(local_id == 0){
        atomic_add(sum,localXs[0]);
    }


    


}