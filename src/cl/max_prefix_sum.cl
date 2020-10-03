#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6


struct Block;

#define WORK_GROUP_SIZE (128)
#define RANGE_PER_WORK_ITEM (2)

__kernel void max_prefix_sum(__global const int* blocks,
                                __global int* maxSum,
                                __global int* index,
                                __global int* newBlocks,
                                __global int* newMaxSum,
                                __global int* newIndex,
                                unsigned int blockSize,
                                bool needInit, unsigned int n)
{
    __local int localBlocks[RANGE_PER_WORK_ITEM * WORK_GROUP_SIZE];
    __local int localMaxSum[RANGE_PER_WORK_ITEM * WORK_GROUP_SIZE];
    __local int localIndex[RANGE_PER_WORK_ITEM * WORK_GROUP_SIZE];
    unsigned int offset = get_group_id(0) * WORK_GROUP_SIZE * RANGE_PER_WORK_ITEM;
    for (unsigned int i = 0; i < RANGE_PER_WORK_ITEM; ++i){
        unsigned int id = i * WORK_GROUP_SIZE + get_local_id(0);
        if (offset + id < n) {
            localBlocks[id] = blocks[offset + id];
            localMaxSum[id] = maxSum[offset + id];
            localIndex[id] = index[offset + id];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int sum = 0, res = 0, ind = 0;
    for (unsigned int i = 0; i < RANGE_PER_WORK_ITEM; ++i) {
        unsigned int x = get_local_id(0) * RANGE_PER_WORK_ITEM + i;
        if (offset + x >= n)
            break;
        if (needInit) {
            if (localBlocks[x] >= 0){
                localMaxSum[x] = localBlocks[x];
                localIndex[x] = 1;
            } else {
                localMaxSum[x] = 0;
                localIndex[x] = 0;
            }
        }
        if (sum + localMaxSum[x] > res){
            res = sum + localMaxSum[x];
            ind = i * blockSize + localIndex[x];
        }
        sum += localBlocks[x];
    }
    newBlocks[get_global_id(0)] = sum;
    newMaxSum[get_global_id(0)] = res;
    newIndex[get_global_id(0)] = ind;
}

// __kernel void max_prefix_sum(__global const int* blocks,
//                                 __global int* maxSum,
//                                 __global int* index,
//                                 __global int* newBlocks,
//                                 __global int* newMaxSum,
//                                 __global int* newIndex,
//                                 unsigned int blockSize,
//                                 bool needInit, unsigned int n)
// {
//     int sum = 0, res = 0, ind = 0;
//     for (unsigned int i = 0; i < RANGE_PER_WORK_ITEM; ++i) {
//         unsigned int x = get_global_id(0) * RANGE_PER_WORK_ITEM + i;
//         if (x >= n)
//             break;
//         if (needInit) {
//             if (blocks[x] >= 0){
//                 maxSum[x] = blocks[x];
//                 index[x] = 1;
//             } else {
//                 maxSum[x] = 0;
//                 index[x] = 0;
//             }
//         }
//         if (sum + maxSum[x] > res){
//             res = sum + maxSum[x];
//             ind = i * blockSize + index[x];
//         }
//         sum += blocks[x];
//     }
//     newBlocks[get_global_id(0)] = sum;
//     newMaxSum[get_global_id(0)] = res;
//     newIndex[get_global_id(0)] = ind;
// }