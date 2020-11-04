#ifndef MERGE_USED_TYPE
    #define MERGE_USED_TYPE float
#endif

typedef MERGE_USED_TYPE mtype;

__kernel void merge(__global mtype* as, __global mtype* as_out, unsigned int n, unsigned int len) {
    unsigned int global_id = get_global_id(0);

    // Return if global_id >= n;
    if (global_id >= n)
        return;

    unsigned int a_start = global_id / (len << 1) * (len << 1);
    unsigned int a_end = min(a_start + len, n);
    unsigned int b_start = a_end;
    unsigned int b_end = min(b_start + len, n);

    // Return if b_start >= n;
    if (b_start >= n) {
        as[global_id] = as_out[global_id];
        return;
    }

    unsigned int a_pos = global_id - a_start;

    //  a->______ *r start
    //  b |     / from the max possible
    // \/ |    /  index lying on diagonal
    //    |   /
    //    |  /
    //    | /
    // *l |/
    // start outside of diagonal
    int l = max((int)a_pos - (int)(b_end - b_start), 0) - 1;
    int r = min(len, a_pos);
    while (l + 1 < r) {
        int m = (l + r) / 2;
        if (as[a_start + m] <= as[b_start + a_pos - m - 1]) {
            l = m;
        } else {
            r = m;
        }
    }

    unsigned int a = a_start + r;
    unsigned int b = b_start + a_pos - r;

    if (a < a_end && (b >= b_end || as[a] <= as[b])) {
        as_out[global_id] = as[a];
    } else {
        as_out[global_id] = as[b];
    }
}
