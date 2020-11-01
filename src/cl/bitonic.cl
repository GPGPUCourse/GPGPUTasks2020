#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

#define WORK_GROUP_SIZE 256

__kernel void bitonic(__global float *as, unsigned int sliceSize, unsigned int step, unsigned int n, unsigned int useLocal) {

    if (useLocal) {

        unsigned int local_id = get_local_id(0);
        unsigned int global_id = get_global_id(0);

        __local float localWorkGroupBuffer[WORK_GROUP_SIZE];

        if (global_id < n) {
            localWorkGroupBuffer[local_id] = as[global_id];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        int flag = global_id % (2 * step) < step;

        while (sliceSize >= 1) {

            unsigned int pos = global_id % (2 * sliceSize);
            int inBounds = global_id + sliceSize < n;

            if (inBounds && pos < sliceSize) {

                float left = localWorkGroupBuffer[local_id];
                float right = localWorkGroupBuffer[local_id + sliceSize];

                if (left > right == flag) {
                    localWorkGroupBuffer[local_id] = right;
                    localWorkGroupBuffer[local_id + sliceSize] = left;
                }

            }

            barrier(CLK_LOCAL_MEM_FENCE);
            sliceSize /= 2;
        }

        if (global_id < n) {
            as[global_id] = localWorkGroupBuffer[local_id];
        }

    } else {

        unsigned int global_id = get_global_id(0);

        int flag = global_id % (2 * step) < step;
        unsigned int pos = global_id % (2 * sliceSize);
        int inBounds = global_id + sliceSize < n;

        if (inBounds && pos < sliceSize) {

            float left = as[global_id];
            float right = as[global_id + sliceSize];

            if (left > right == flag) {

                as[global_id] = right;
                as[global_id + sliceSize] = left;
            }
        }
    }
}
