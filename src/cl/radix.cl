#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 128
__kernel void radix_sum(__global unsigned int* as, unsigned int shift, int N, __global unsigned int* bucket_cnt) {
    const unsigned int id = get_global_id(0);
    const unsigned int local_id = get_local_id(0);

    const unsigned int mask = 8 + 4 + 2 + 1;
    __local unsigned int local_a[WORK_GROUP_SIZE];
    if (id < N) {
        local_a[local_id] = as[id];
    } else {
        local_a[local_id] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id == 0) {
        for (int i = 0; i < 16; ++i) {
            bucket_cnt[(id/WORK_GROUP_SIZE) * 16 + i] = 0;
        }

        for (int i = 0; i < WORK_GROUP_SIZE; ++i) {
            if (id + i < N) {
                bucket_cnt[(id/WORK_GROUP_SIZE) * 16 + ((local_a[i] >> shift) & mask)] += 1;
            }
        }
    }
}

#define WORK_GROUP_SIZE 128
__kernel void radix_exchange(const __global unsigned int* as_gpu, __global unsigned int* as_gpu_out,
                                  unsigned int shift, int n, __global unsigned int*  prefix_sum,
                                  __global unsigned int* bucket_prefix_sum) {
    const unsigned int id = get_global_id(0);
    const unsigned int mask = 8 + 4 + 2 + 1;
    if (id < n) {
        unsigned int val = ((as_gpu[id] >> shift)&mask);
        int sm_before = 0;
        for (int i = 0; i < val; ++i) {
            sm_before += bucket_prefix_sum[((n - 1)/WORK_GROUP_SIZE)*16 + i];
        }

        const unsigned int new_id = sm_before + prefix_sum[id];
        as_gpu_out[new_id] = as_gpu[id];
    }
}

#define WORK_GROUP_SIZE 128
__kernel void bucket_cal_prefix_sum(__global unsigned int* bucket_sum, unsigned int step, unsigned int bucket_cnt) {
    const unsigned int id = get_global_id(0);
    const unsigned int local_id = get_local_id(0);
    unsigned int pos_sum = ((id + 1)* step - 1) * 16;
    __local unsigned int local_a[WORK_GROUP_SIZE];
    for (int i = 0; i < 16; ++i) {
        if (pos_sum + i < bucket_cnt * 16) {
            local_a[local_id] = bucket_sum[pos_sum + i];
        } else {
            local_a[local_id] = 0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (local_id == 0) {
           unsigned int sum = 0;
           for (int j = 0; j < WORK_GROUP_SIZE; ++j) {
               local_a[j] += sum;
               sum = local_a[j];
           }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (pos_sum + i < bucket_cnt * 16) {
            bucket_sum[pos_sum + i] = local_a[local_id];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

#define WORK_GROUP_SIZE 128
__kernel void prefix_sum_calc(__global unsigned int* as, unsigned int shift, int N, __global unsigned int* bucket_sum,  __global unsigned int* prefix_sum) {
    const unsigned int id = get_global_id(0);
    const unsigned int local_id = get_local_id(0);

    const unsigned int mask = 8 + 4 + 2 + 1;
    unsigned int val = 0;
    if (id < N) {
        val = ((as[id] >> shift) & mask);
    }
    unsigned int sum = 0;
    if ((id/WORK_GROUP_SIZE) > 0 && id < N) {
        sum = bucket_sum[(id/WORK_GROUP_SIZE - 1) * 16 + val];
    }

    __local unsigned int local_a[WORK_GROUP_SIZE];
    __local unsigned int local_sum[16];

    if (id < N) {
       local_a[local_id] = as[id];
    } else {
       local_a[local_id] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id == 0) {
        for (int i = 0; i < 16; ++i) {
            local_sum[i] = 0;
        }

        for (int i = 0; i < WORK_GROUP_SIZE; ++i) {
            unsigned int lc_val = ((local_a[i] >> shift) & mask);
            local_a[i] = local_sum[lc_val];
            if (id + i < N) {
                local_sum[lc_val] += 1;
            }
        }
     }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (id < N) {
        prefix_sum[id] = local_a[local_id] + sum;
    }
}

#define WORK_GROUP_SIZE 128
__kernel void recalc_prefix_sum(__global unsigned int* bucket_sum, unsigned int step, unsigned int bucket_cnt) {
    const unsigned int id = get_global_id(0);
    const unsigned int local_id = get_local_id(0);
    unsigned int pos_sum = ((id + 1) * step - 1) * 16;

    if (pos_sum < bucket_cnt * 16) {
        for (int i = 0; i < 16; ++i) {
            unsigned int sum = 0;
            if (id - local_id > 0) {
                   sum = bucket_sum[((id - local_id)*step - 1)*16 + i];
            }

            if (local_id != WORK_GROUP_SIZE - 1) {
                bucket_sum[pos_sum + i] += sum;
            }
        }
    }
}

