#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

#define WORK_GROUP_SIZE 128

__kernel void addBuffer(__global unsigned int *as, __global unsigned int *buff,
                        unsigned int aSize) {

    unsigned int global_id = get_global_id(0);
    unsigned int group_id = get_group_id(0);

    if (group_id > 0 && global_id < aSize) {

        as[global_id] += buff[group_id - 1];
    }
}

__kernel void calculatePrefix(__global unsigned int *as, __global unsigned int *buff, unsigned int aSize) {

    unsigned int local_id = get_local_id(0);
    unsigned int global_id = get_global_id(0);

    __local unsigned int localWorkGroupBuffer[WORK_GROUP_SIZE];
    __local unsigned int summatorDoubleBufferTree[2 * WORK_GROUP_SIZE];

    int inBounds = global_id < aSize;
    localWorkGroupBuffer[local_id] = inBounds ? as[global_id] : 0;

    summatorDoubleBufferTree[local_id + WORK_GROUP_SIZE] = inBounds ? localWorkGroupBuffer[local_id] : 0;

    summatorDoubleBufferTree[local_id] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int currentSlice = WORK_GROUP_SIZE / 2; currentSlice > 0; currentSlice /= 2) {

        if (local_id < currentSlice) {

            unsigned int succId = (local_id + currentSlice) * 2;
            unsigned int ssuccId = succId + 1;

            summatorDoubleBufferTree[local_id + currentSlice] =
                    summatorDoubleBufferTree[succId] + summatorDoubleBufferTree[ssuccId];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {

        summatorDoubleBufferTree[1] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int currentSliceI = 1; currentSliceI < WORK_GROUP_SIZE; currentSliceI *= 2) {

        unsigned int succId = 2 * (local_id + currentSliceI);
        unsigned int ssuccId = succId + 1;

        if (local_id < currentSliceI) {

            summatorDoubleBufferTree[ssuccId] =
                    summatorDoubleBufferTree[succId] + summatorDoubleBufferTree[local_id + currentSliceI];
            summatorDoubleBufferTree[succId] = summatorDoubleBufferTree[local_id + currentSliceI];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id == 0) {

        buff[get_group_id(0)] =
                localWorkGroupBuffer[WORK_GROUP_SIZE - 1] + summatorDoubleBufferTree[2 * WORK_GROUP_SIZE - 1];
    }

    as[global_id] = localWorkGroupBuffer[local_id] + summatorDoubleBufferTree[local_id + WORK_GROUP_SIZE];
}

__kernel void radixByBits(__global unsigned int *as, __global unsigned int *pos0,
                          __global unsigned int *pos1, unsigned int curr, unsigned int n) {

    unsigned int global_id = get_global_id(0);

    if (global_id < n) {

        unsigned int current = as[global_id];
        int BitIsEqual1 = (current >> curr) & 1;

        pos1[global_id] = BitIsEqual1 ? 1 : 0;
        pos0[global_id] = BitIsEqual1 ? 0 : 1;
    }
}

__kernel void radix(__global unsigned int *as, __global unsigned int *dest,
                    __global unsigned int *pos0, __global unsigned int *pos1,
                    unsigned int curr, unsigned int n) {

    unsigned int global_id = get_global_id(0);

    if (global_id < n) {

        unsigned int current = as[global_id];

        int bitIsEqual1 = (current >> curr) & 1;

        if (bitIsEqual1) {

            dest[pos0[n - 1] + pos1[global_id] - 1] = current;
        } else {

            dest[pos0[global_id] - 1] = current;
        }
    }
}
