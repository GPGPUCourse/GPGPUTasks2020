__kernel void merge(__global const float* as,
                    __global float* as_out,
                    unsigned int sorted_size,
                    unsigned int n)
{
    const unsigned int id = get_global_id(0);
    if (id >= n) {
        return;
    }

    const unsigned int first_start = (id / (2 * sorted_size)) * 2 * sorted_size;
    const unsigned int second_start = first_start + sorted_size;
    const unsigned int second_size = min(sorted_size,  n - second_start);
    if (second_start >= n) {
        as_out[id] = as[id];
        return;
    }

    const int is_left = id < second_start;
    // we are looking lower bound and upper bound to find our position in merged array
    // if item is from left array find  #{elems in right} < item: lower bound
    // if item is from right array find #{elems in left} <= item: upper bound
    int lower = is_left ? second_start : first_start;
    int upper = is_left ? (second_start + second_size) : second_start;

    const float item = as[id];

    while (lower < upper) {
        int m = (lower + upper) / 2;
        if (as[m] < item) {
            lower = m + 1;
        } else {
            upper = m;
        }
    }
    upper = lower;
    while (upper < second_start && as[upper] == item) {
        ++upper;
    }

    const int left_offset = is_left ? id : upper;
    const int right_offset = (is_left ? lower : id) - second_start;

    as_out[left_offset + right_offset] = item;
}
