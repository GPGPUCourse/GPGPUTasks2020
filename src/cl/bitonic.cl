#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define LOWEST_LEVEL 7


__kernel void bitonic(__global float* as,
                        unsigned int currentLevel,
                        unsigned int internalLevel,
                        unsigned int n)
{
    if (currentLevel == LOWEST_LEVEL){
        __local float cache[1 << (LOWEST_LEVEL + 1)];

        unsigned int shift = (get_global_id(0) >> currentLevel) << currentLevel;
        cache[get_local_id(0)] = as[get_global_id(0) + shift];
        if (get_global_id(0) + shift + (1 << currentLevel) < n)
            cache[get_local_id(0) + (1 << currentLevel)] = as[get_global_id(0) + shift + (1 << currentLevel)];

        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = 0; i <= LOWEST_LEVEL; i++){
            for (int j = 0; j <= i; j++) {

                unsigned int blockID = (get_local_id(0) >> i);
                unsigned int internalBlockID = (get_local_id(0) - (blockID << i)) >> (i - j);
                unsigned int l = blockID * (1 << i) + internalBlockID * (1 << (i - j)) + get_local_id(0);
                unsigned int r = l + (1 << (i - j));
                if (r + get_global_id(0) - get_local_id(0) < n) {
                    bool up = (get_global_id(0) >> i) & 1;
                    if ((up && (cache[l] < cache[r])) || (!up && (cache[l] > cache[r]))){
                        float x = cache[l];
                        cache[l] = cache[r];
                        cache[r] = x;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }

        as[get_global_id(0) + shift] = cache[get_local_id(0)];
        if (get_global_id(0) + (1 << currentLevel) < n)
            as[get_global_id(0) + shift + (1 << currentLevel)] = cache[get_local_id(0) + (1 << currentLevel)];
    } else {
        unsigned int blockID = get_global_id(0) >> currentLevel;
        unsigned int internalBlockID = (get_global_id(0) - (blockID << currentLevel)) >> (currentLevel - internalLevel);
        unsigned int l = get_global_id(0) + (blockID << currentLevel) + (internalBlockID << (currentLevel - internalLevel));
        unsigned int r = l + (1 << (currentLevel - internalLevel));
        if (r < n) {
            float L = as[l];
            float R = as[r];
            if (((blockID & 1) && (L < R)) || (!(blockID & 1) && (L > R))){
                float x = L;
                as[l] = R;
                as[r] = x;
            }
        }
    }
}
