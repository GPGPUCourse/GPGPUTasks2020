#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define DATA_PER_WORKITEM 8
#define WORK_GROUP_SIZE 128

__kernel void sum_less_atomic(__global const unsigned int *sum_buffer, const unsigned int n, __global unsigned int *result) {

    const unsigned int local_id = get_local_id(0);
    __local unsigned int local_buffer[WORK_GROUP_SIZE];

    unsigned int WG_SUM = 0;

    // Every work-item gets at minimum 8 pairs of elements
    for (unsigned int iter_wi_data = 0; iter_wi_data < DATA_PER_WORKITEM; ++iter_wi_data) {
        const unsigned int global_id = DATA_PER_WORKITEM * WORK_GROUP_SIZE * get_group_id(0) + // workgroup offset
                                        WORK_GROUP_SIZE * iter_wi_data  + // offset using current iteration
                                        local_id; // offset by local_id

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

        if(local_id == 0)
            WG_SUM += local_buffer[0];
    }

    if(local_id == 0)
        atomic_add(result, WG_SUM);
}