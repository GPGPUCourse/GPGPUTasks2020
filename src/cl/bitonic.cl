#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

__kernel void bitonic(__global float* as, unsigned int n, unsigned int block, unsigned int sub_block)
{
    int id = get_global_id(0);
    
    if (block == 0) { // evaluate blocks from 2 to WORK_GROUP_SIZE
        int lid = get_local_id(0);
        local float A[WORK_GROUP_SIZE];
        
        A[lid] = as[id];
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (int i = 2; i <= WORK_GROUP_SIZE; i *= 2) {
            for (int j = i / 2; j >= 1; j /= 2) {
                int to = id ^ j;
                if (id < to) {
                    int lto = to % WORK_GROUP_SIZE;
                    bool decreasing = (i & id) != 0;
                    if ((decreasing && A[lid] < A[lto]) || (!decreasing && A[lid] > A[lto]))
                    {
                        float tmp = A[lid];
                        A[lid] = A[lto];
                        A[lto] = tmp;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
        
        as[id] = A[lid];
    }
    
    else { // block is bigger then WORK_GROUP_SIZE
        int i = block;
        int j = sub_block;
        
        int to = id ^ j;
        if (id < to) {
            bool decreasing = (i & id) != 0;
            if ((decreasing && as[id] < as[to]) || (!decreasing && as[id] > as[to]))
            {
                float tmp = as[id];
                as[id] = as[to];
                as[to] = tmp;
            }
        }
    }
}
