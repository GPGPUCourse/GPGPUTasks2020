#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

#define WORK_GROUP_SIZE 256
#define WORK_GROUP_BITS 8
__kernel void max_prefix_sum(__global const int *as,
                             __global const int *ps,
                             __global const unsigned int *results,
                             __global int *out_as,
                             __global int *out_ps,
                             __global unsigned int *out_results,
                             unsigned int n) {
    unsigned int globalId = get_global_id(0);
    unsigned int localId = get_local_id(0);
    __local int local_xs[WORK_GROUP_SIZE];
    __local int local_ps[WORK_GROUP_SIZE];
    __local unsigned int local_results[WORK_GROUP_SIZE];
//    butterfly transform
    unsigned int idx = 0;
    for (int i = 0; i < WORK_GROUP_BITS; ++i) {
        idx = (idx << 1) + ((localId >> i) & 1);
    }
    local_xs[idx] = (globalId < n ? as[globalId] : 0);
    local_ps[idx] = (globalId < n ? ps[globalId] : 0);
    local_results[idx] = (globalId < n ? results[globalId] : 0);
    barrier(CLK_LOCAL_MEM_FENCE);
//    now on each step we look onto neighbours in `as`
    for (unsigned int block = WORK_GROUP_SIZE; block > 1; block /= 2) {
        if (2 * localId < block) {
            int ls = local_xs[localId];
            int rs = local_xs[localId + block / 2];
            int lps = local_ps[localId];
            int rps = local_ps[localId + block / 2];
            unsigned int lres = local_results[localId];
            unsigned int rres = local_results[localId + block / 2];
            local_xs[localId] = ls + rs;
            if (ls + rps > lps) {
                local_ps[localId] = ls + rps;
                local_results[localId] = rres;
            } else {
                local_ps[localId] = lps;
                local_results[localId] = lres;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (localId == 0) {
        out_as[globalId / WORK_GROUP_SIZE] = local_xs[0];
        out_ps[globalId / WORK_GROUP_SIZE] = local_ps[0];
        out_results[globalId / WORK_GROUP_SIZE] = local_results[0];
    }
}