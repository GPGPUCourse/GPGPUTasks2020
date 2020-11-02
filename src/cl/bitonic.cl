#define BLOCK_SIZE 64

__kernel void bitonic(__global float* as, int run, int step, int subBlockSize)
{
    int blockSize = 1 << run;
    int numOfBlocks = get_global_size(0) / blockSize;

    int blockIndex = get_global_id(0) / blockSize;

    int blockFirstItemId = blockIndex * blockSize;
    int odd = (blockIndex & 1) == 1;
    int itemLocalId = get_global_id(0) - blockFirstItemId;

    int subBlockMiddle = subBlockSize / 2;

    bool allowed = (itemLocalId % subBlockSize) < subBlockMiddle;

    if(allowed) {
        int bDataIndex = 0;
        int aDataIndex = 0;
        if (odd) {

            int subBlockId = get_global_id(0) / subBlockSize;
            int offsetToSubBlock = subBlockId * subBlockSize;

            int idx = get_global_id(0) - offsetToSubBlock;

            bDataIndex = offsetToSubBlock + idx;
            aDataIndex = bDataIndex + step;

        } else {
            aDataIndex = get_global_id(0);
            bDataIndex = aDataIndex + step;
        }

        float a = as[aDataIndex];
        float b = as[bDataIndex];

        if(a > b) {
            as[aDataIndex] = b;
            as[bDataIndex] = a;
        }
    }
}