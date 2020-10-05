#define WG_SIZE 32

__kernel void sum(__global int* as,
                  __global int* result,
                  __global int* maxPrefixPerGroup,
                  __global int* maxPrefixPerGroupIndex)
{
    __local int groupLine[WG_SIZE];
    groupLine[get_local_id(0)] = as[get_global_id(0)];
    barrier(CLK_LOCAL_MEM_FENCE);

    if(get_local_id(0) == 0) {
        int ac = 0;
        int maxPrefix = 0;
        int maxIndex = 0;
        int groupSize = get_local_size(0);
        for(int i = 0; i < groupSize; i++) {
            int element = groupLine[i];
            int elementIndex = get_group_id(0) * get_local_size(0) + i;

            ac += element;
            if (ac > maxPrefix) {
                maxPrefix = ac;
                maxIndex = elementIndex;
            }
        }
        result[get_group_id(0)] = ac;
        maxPrefixPerGroup[get_group_id(0)] = maxPrefix;
        maxPrefixPerGroupIndex[get_group_id(0)] = maxIndex;
    }
}