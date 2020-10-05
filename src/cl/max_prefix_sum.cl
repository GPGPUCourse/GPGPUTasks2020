//TODO
#define WORK_GROUP_SIZE 128
#define PER_ITEM 32

__kernel void getSums(__global const unsigned int *xs,
                        unsigned int n,
                        __global unsigned int *sums) {
    unsigned int sum = 0;

    if (get_local_id(0) == 0) {
        for (unsigned int i = 0; i < n; ++i) {
            sum += xs[i];
            sums[i] = sum;
        }
    }
}
