__kernel void merge(__global float* as,
                    __global float* buf,
                    int n, int l) {

    const unsigned int id = get_global_id(0);
    if (id >= n)
        return;

    // initialize box margins (horizontal/vertical)
    int size = 2 * l;
    int fr_v = id / size * size, to_v = min(n, fr_v + l);
    // check if out the box
    if (fr_v + l >= n) {
        as[id] = buf[id];
        return;
    }
    int fr_h = to_v, to_h = min(n, fr_h + l);
//    printf("id=%d fr_v-%d to_v=%d fr_h=%d to_h=%d\n", id, fr_v, to_v, fr_h, to_h);

    // find diagonal idx-s
    int idx = id - fr_v;
    int left = max(0, idx - (to_h - fr_h)) - 1;
    int right = min(idx, l);
//    printf("idx=%d left-%d right=%d\n", idx, left, right);

    // binary search through diagonal el-s
    while (left < right - 1) {

        int mid = (left + right) >> 1;
        if (as[fr_v+mid] <= as[fr_h-mid+idx-1])
            left = mid;
        else
            right = mid;
    }
    // 1 -> 0 transfer idx-s
    int found_v = fr_v + right;
    int found_h = fr_h + idx - right;
    // place minimum from corresponding numbers
    bool less = found_v < to_v && as[found_v] <= as[found_h];
    bool out = found_v < to_v && found_h >= to_h;
    buf[id] =  (less || out) ? as[found_v] : as[found_h];
}