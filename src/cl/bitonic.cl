#define BLOCK_SIZE 32

__kernel void bitonicLocal(__global float* as)
{
    __local float numbers[BLOCK_SIZE];

    numbers[get_local_id(0)] = as[get_global_id(0)];
    barrier(CLK_LOCAL_MEM_FENCE);
    int maxRuns = int(log2(float(get_local_size(0))));

    for(int run = 1; run < maxRuns; run++) {

        int blockSize = 1 << run;

        int step = blockSize / 2;

        int blockIndex = get_local_id(0) / blockSize;

        int blockLocalId = blockIndex * blockSize;

        int subBlocksCount = 1;
        int subBlockSize = blockSize;

        int odd = (blockIndex & 1) == 1;

        while(step > 0) {

            int itemLocalId = get_local_id(0) - blockLocalId;
            int subBlockMiddle = subBlockSize / 2;

            bool allowed = (itemLocalId % subBlockSize) < subBlockMiddle;

            if(allowed) {
                int bDataIndex = 0;
                int aDataIndex = 0;
                if (odd) {
                    bDataIndex = get_local_id(0);
                    aDataIndex = bDataIndex + step;

                } else {
                    aDataIndex = blockLocalId + itemLocalId;
                    bDataIndex = aDataIndex + step;
                }

                float a = numbers[aDataIndex];
                float b = numbers[bDataIndex];

                if(a > b) {
                    numbers[aDataIndex] = b;
                    numbers[bDataIndex] = a;
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
            step >>= 1;
            subBlockSize >>= 1;
        }
    }
    as[get_global_id(0)] = numbers[get_local_id(0)];
}

__kernel void bitonicGlobal(__global float* as, int run, int step, int subBlockSize)
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
            bDataIndex = get_global_id(0);
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

__kernel void bitonicStep(__global float* as, int blockSize, int subBlockSize)
{
    __local float numbers[BLOCK_SIZE];

    numbers[get_local_id(0)] = as[get_global_id(0)];
    barrier(CLK_LOCAL_MEM_FENCE);

    int step = subBlockSize / 2;

    int blockIndex = get_global_id(0) / blockSize;
    int odd = (blockIndex & 1) == 1;

    while(step > 0) {

        int blockLocalId = (get_local_id(0) / subBlockSize) * subBlockSize;
        int itemLocalId = get_local_id(0) - blockLocalId;
        int subBlockMiddle = subBlockSize / 2;

        bool allowed = (itemLocalId % subBlockSize) < subBlockMiddle;

        if(allowed) {
            int bDataIndex = 0;
            int aDataIndex = 0;
            if (odd) {
                bDataIndex = get_local_id(0);
                aDataIndex = bDataIndex + step;

            } else {
                aDataIndex = get_local_id(0);
                bDataIndex = aDataIndex + step;
            }

            float a = numbers[aDataIndex];
            float b = numbers[bDataIndex];

            if(a > b) {
                numbers[aDataIndex] = b;
                numbers[bDataIndex] = a;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        step >>= 1;
        subBlockSize >>= 1;
    }

    as[get_global_id(0)] = numbers[get_local_id(0)];
}