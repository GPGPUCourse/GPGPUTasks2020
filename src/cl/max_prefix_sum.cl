#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#define VALS_IN_STEP 16
#define WORKGROUP_SIZE 128
#define INT_MIN -1000000
#endif

#line 8

#define LOCAL_SUM_TOGETHER 0
#define NO_LOCAL 1

#define KERNEL_VERSION LOCAL_SUM_TOGETHER

#define PRINT_DEBUG \
if (loc_i == 0) {                                            \
    printf("--- %d ---", workgroup_inp_start);               \
    printf("\nlocal_maxpref_i: ");                           \
    for (int i = 0; i < VALS_IN_STEP*WORKGROUP_SIZE && i < current_n; ++i) {  \
        printf("%d ", local_maxpref_i[i]);                   \
    }                                                        \
    printf("\nlocal_maxpref_v: ");                           \
    for (int i = 0; i < VALS_IN_STEP*WORKGROUP_SIZE && i < current_n; ++i) {  \
        printf("%d ", local_maxpref_v[i]);                   \
    }                                                        \
    printf("\nlocal_sum:       ");                           \
    for (int i = 0; i < VALS_IN_STEP*WORKGROUP_SIZE && i < current_n; ++i) {  \
        printf("%d ", local_sum[i]);                         \
    }                                                        \
    printf("\n");                                            \
}

#if KERNEL_VERSION==LOCAL_SUM_TOGETHER
__kernel void max_prefix_sum(
    __global const unsigned int* inp_maxpref_i, // current_n
    __global const int* inp_maxpref_v, // current_n
    __global const int* inp_sum, // current_n
    __global unsigned int* out_maxpref_i, // next_n
    __global int* out_maxpref_v, // next_n
    __global int* out_sum,// next_n
    unsigned int current_n,
    unsigned int next_n)
{
    unsigned int localId = get_local_id(0);
    unsigned int loc_i = get_global_id(0);

    __local unsigned int local_maxpref_i[(VALS_IN_STEP+1)*WORKGROUP_SIZE];
    __local int local_maxpref_v[(VALS_IN_STEP+1)*WORKGROUP_SIZE];
    __local int local_sum[(VALS_IN_STEP+1)*WORKGROUP_SIZE];
    unsigned int workgroup_inp_start = (loc_i-localId)*VALS_IN_STEP;

    for (int i = 0; i < VALS_IN_STEP; ++i) {
        unsigned inp_shift = i*WORKGROUP_SIZE + localId;
        unsigned idx = workgroup_inp_start + inp_shift;
        unsigned local_idx = inp_shift + inp_shift/VALS_IN_STEP;
        if (idx < current_n) {
            local_maxpref_i[local_idx] = inp_maxpref_i[idx];
            local_maxpref_v[local_idx] = inp_maxpref_v[idx];
            local_sum[local_idx] = inp_sum[idx];
        } else {
            local_maxpref_i[local_idx] = 0;
            local_maxpref_v[local_idx] = 0;
            local_sum[local_idx] = 0;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    //if (loc_i == 0) PRINT_DEBUG

    unsigned int thread_start = localId * (VALS_IN_STEP+1);
    int maxpref_v = INT_MIN;
    unsigned int maxpref_i = -1;
    int sum = 0;
    int sum_prev = 0;
    for (int i = thread_start; i < thread_start + VALS_IN_STEP; ++i) {
        sum = local_maxpref_v[i] + sum_prev;
        sum_prev += local_sum[i];
        if (sum > maxpref_v) {
            maxpref_v = sum;
            maxpref_i = local_maxpref_i[i];
        }
    }
    //if (loc_i < current_n) printf("wrote in %d: %d %d %d\n", loc_i, maxpref_i, maxpref_v, sum_prev);
    out_maxpref_i[loc_i] = maxpref_i;
    out_maxpref_v[loc_i] = maxpref_v;
    out_sum[loc_i] = sum_prev;
}
#endif

#if KERNEL_VERSION==NO_LOCAL
__kernel void max_prefix_sum(
    __global const unsigned int* inp_maxpref_i, // current_n
    __global const int* inp_maxpref_v, // current_n
    __global const int* inp_sum, // current_n
    __global unsigned int* out_maxpref_i, // next_n
    __global int* out_maxpref_v, // next_n
    __global int* out_sum,// next_n
    unsigned int current_n,
    unsigned int next_n)
{
    unsigned int localId = get_local_id(0);
    unsigned int loc_i = get_global_id(0);

    unsigned int thread_start = loc_i * (VALS_IN_STEP);
    int maxpref_v = INT_MIN;
    unsigned int maxpref_i = -1;
    int sum = 0;
    int sum_prev = 0;
    for (int i = thread_start; i < thread_start + VALS_IN_STEP && i < current_n; ++i) {
        sum = inp_maxpref_v[i] + sum_prev;
        sum_prev += inp_sum[i];
        #if 1
        if (sum > maxpref_v) {
            maxpref_v = sum;
            maxpref_i = inp_maxpref_i[i];
        }
        #else
        int cond = sum > maxpref_v;
        maxpref_v = cond*sum + (1-cond)*maxpref_v;
        maxpref_i = cond*local_maxpref_i[i] + (1-cond)*maxpref_i;
        #endif
    }
    //if (loc_i < current_n) printf("wrote in %d: %d %d %d\n", loc_i, maxpref_i, maxpref_v, sum_prev);
    out_maxpref_i[loc_i] = maxpref_i;
    out_maxpref_v[loc_i] = maxpref_v;
    out_sum[loc_i] = sum_prev;
}
#endif
