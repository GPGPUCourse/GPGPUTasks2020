#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void merge(const __global float * as,  __global float * as_out, int sorted_len, int n) {
    //printf("merge start\n");
    const int id = get_global_id(0);
    int blc_start = (id/(2*sorted_len))*2*sorted_len;
    int sc_blc = blc_start + sorted_len;
    int blc_id = id - blc_start;
    int lft = 0, rgh = blc_id  + 2;
    //printf("%d %d %d %d\n", id, blc_start, sc_blc, rgh);
    while (lft + 1 < rgh) {
        int md = (lft + rgh)/2;

        if (md > sorted_len) {
           rgh = md;
        } else if (blc_id  - md + 1 > sorted_len) {
            lft = md;
        } else if (md - 1 < 0 || (blc_id - md >= 0 && as[blc_start + md - 1] <= as[sc_blc + blc_id  - md])) {
            lft = md;
        } else {
            rgh = md;
        }
    }

    if (lft + sorted_len < blc_id + 1) {
        as_out[id] = as[blc_start + blc_id - sorted_len];
    } else if (blc_id + 1 > sorted_len && as[sc_blc - 1] < as[sc_blc]) {
        as_out[id] = as[id];
    } else if (lft < sorted_len && as[blc_start + lft] < as[sc_blc + blc_id  - lft]) {
        as_out[id] = as[blc_start + lft];
    } else {
        as_out[id] = as[sc_blc + blc_id  - lft];
    }
}