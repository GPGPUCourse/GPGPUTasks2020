__kernel void mergePath(__global float* as,
                        __global float* bs,
                        __global int* pathPoints,
                        int aSize,
                        int step) {

    int frameSize = aSize;
    int aIndex = 0;
    int bIndex = 0;

    bool topPart = get_global_id(0) < get_global_size(0) / 2;

    int groupIndexOffset = get_group_id(0) * get_local_size(0);

    // A
    int aTopIndex = 0;
    int aBottomIndex = (get_local_id(0) % (get_global_size(0) / 2) + 1);
    aBottomIndex += groupIndexOffset;

    if(!topPart) {
        aTopIndex = get_global_id(0) % (get_global_size(0) / 2);
        aBottomIndex = (get_global_size(0) / 2);
    }

    int aTop = aTopIndex * step + aIndex;
    int aBottom = (aBottomIndex * step) + aIndex;

    // B
    int bTopIndex = 0;
    int bBottomIndex = (get_local_id(0) % (get_global_size(0) / 2) + 1);
    bBottomIndex += groupIndexOffset;
    if(!topPart) {
        bTopIndex = get_global_id(0) % (get_global_size(0) / 2);
        bBottomIndex = (get_global_size(0) / 2);
    }

    int bTop = bIndex + (bTopIndex * step);
    int bBottom = (bBottomIndex * step) + bIndex;

    int midf = float(aBottom - aTop) / 2.0;
    int mid = (aBottom - aTop) / 2;

    while(mid > 0) {

        int aI = aTop + mid;
        int bI = bTop + mid;

        if (as[aI] > bs[bI - 1]) {
            if (as[aI - 1] <= bs[bI]) {
                pathPoints[(get_global_id(0)  + 1) * 2] = aI;
                pathPoints[(get_global_id(0)  + 1) * 2 + 1] = bI;
                break;
            } else {
                aBottom = aI + 1;
                bTop = bI - 1;
            }
        } else {
            aTop = aI - 1;
        }

        midf = ceil(float(aBottom - aTop) / 2.0);
        mid = int(midf);

        if(mid == 1) {
            pathPoints[(get_global_id(0) + 1) * 2] = aI;
            pathPoints[(get_global_id(0) + 1) * 2 + 1] = bI;

            break;
        }
    }
}


__kernel void mergeBlock(__global float* as,
                         __global float* bs,
                         __global float* merged,
                         __global int* pathPoints,
                         int aSize) {

    int aIndex = pathPoints[get_global_id(0) * 2];
    int bIndex = pathPoints[get_global_id(0) * 2 + 1];

    int nextAIndex = min(pathPoints[get_global_id(0) * 2 + 2], aSize);
    int nextBIndex = min(pathPoints[get_global_id(0) * 2 + 3], aSize);


    int count = 0;
    while((aIndex < nextAIndex || bIndex < nextBIndex) && count++ < 1000) {

        float a = as[aIndex];
        float b = bs[bIndex];

        if((a < b && aIndex < aSize) || bIndex >= aSize) {
            merged[aIndex + bIndex] = a;
            aIndex++;
        } else {
            merged[aIndex + bIndex] = b;
            bIndex++;
        }
    }
}