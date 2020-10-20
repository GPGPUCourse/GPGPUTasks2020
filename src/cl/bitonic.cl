#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 256

__kernel void bitonic(__global float* as, unsigned int offset, unsigned int n, unsigned int first_it)
{
    int local_id = get_local_id(0);
    int global_id = get_global_id(0);
    __local float memory[WORK_GROUP_SIZE];

    if (offset >= WORK_GROUP_SIZE) {
        int part_number = global_id / offset;

        if (part_number % 2 != 1 && global_id + offset < n) {
            if (!first_it) {
                float a = as[global_id];
                float b = as[global_id + offset];
                as[global_id] = min(a, b);
                as[global_id + offset] = max(a, b);
            } else {
                float a = as[global_id];
                float b = as[global_id + 2 * offset - global_id % offset * 2 - 1];
                as[global_id] = min(a, b);
                as[global_id + 2 * offset - global_id % offset * 2 - 1] = max(a, b);
            }
        }
    } else {
        if (global_id < n) {
            memory[local_id] = as[global_id];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        int part_number = 0;

        while (offset > 0) {
            part_number = local_id / offset;

            if (part_number % 2 != 1 && global_id + offset < n) {
                if (!first_it) {
                    float a = memory[local_id];
                    float b = memory[local_id + offset];
                    memory[local_id] = min(a, b);
                    memory[local_id + offset] = max(a, b);
                } else {
                    float a = memory[local_id];
                    float b = memory[local_id + 2 * offset - local_id % offset * 2 - 1];
                    memory[local_id] = min(a, b);
                    memory[local_id + 2 * offset - local_id % offset * 2 - 1] = max(a, b);
                }
            }
            first_it = 0;
            offset /= 2;
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        as[global_id] = memory[local_id];
    }
}
