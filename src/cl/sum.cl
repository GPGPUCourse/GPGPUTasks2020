#define WG_SIZE 32

__kernel void sum(__global unsigned int* as,
                  __global unsigned int* result)
{
    __local unsigned int lineToSum[WG_SIZE + WG_SIZE];
    __local unsigned int lineSum[WG_SIZE];

    int groupDataIndex = get_group_id(0) * 2 * get_local_size(0);

    int elementAIdx = get_local_id(0);
    int elementBIdx = get_local_id(0) + get_local_size(0);

    int itemAToAddIdx = groupDataIndex + elementAIdx;
    int itemBToAddIdx = groupDataIndex + elementBIdx;

    lineToSum[elementAIdx] = as[itemAToAddIdx];
    lineToSum[elementBIdx] = as[itemBToAddIdx];

    lineSum[elementAIdx] = lineToSum[elementAIdx] + lineToSum[elementBIdx];

    barrier(CLK_LOCAL_MEM_FENCE);

    if(get_local_id(0) == 0) {
        unsigned int ac = 0;
        unsigned int groupSize = get_local_size(0);
        for(unsigned int i = 0; i < groupSize; i++) {
            ac += lineSum[i];
        }
        result[get_group_id(0)] = ac;
    }
}