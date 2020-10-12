#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 128
__kernel void sum_recursive(__global const unsigned int *sum_buffer,
                            __global unsigned int *swap_sum_buffer,
                            const unsigned int n) {

    const unsigned int global_id = get_global_id(0);
    const unsigned int local_id = get_local_id(0);

    __local unsigned int local_buffer[WORK_GROUP_SIZE];

    if (global_id < n){ // copy from VRAM to local memory
        local_buffer[local_id] = sum_buffer[global_id];
    }
    else{ // pad for work group alignment
        local_buffer[local_id] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE); // wait for local memory update
    for (int nvalues = WORK_GROUP_SIZE; nvalues > 1; nvalues /= 2) {
        if(2 * local_id < nvalues){
            unsigned int first = local_buffer[local_id];
            unsigned int second = local_buffer[local_id + (nvalues / 2)];
            local_buffer[local_id] = first + second;
        }
        barrier(CLK_LOCAL_MEM_FENCE); // wait for tree summation update
    }

    if(local_id == 0){
        swap_sum_buffer[get_group_id(0)] = local_buffer[0];
    }

}