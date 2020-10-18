__kernel void wikipedia_bitonic(__global float* as,
                                unsigned int max_strip,
                                unsigned int strip,
                                unsigned int n)
{
    // total number of work items is n
    const unsigned int index = get_global_id(0);
    if (index >= n) {
        return;
    }
    const unsigned int step = strip / 2;

    unsigned int left_idx = index;
    unsigned int right_idx = index ^ strip;
    const float is_blue = (left_idx & max_strip) != 0 ? 1.f : 0.f;

    // point of code divergence
    // half of work items should stop here
    if (left_idx <= right_idx) {
        return;
    }

    const float left = as[left_idx];
    const float right = as[right_idx];

    as[left_idx] = is_blue * min(left, right) + (1.f - is_blue) * max(left, right);
    as[right_idx] = is_blue * max(left, right) + (1.f - is_blue) * min(left, right);
}


__kernel void bitonic(__global float* as,
                      unsigned int global_strip,
                      unsigned int local_strip,
                      unsigned int n)
{
    // this version fixes code divergence
    // total number of work items is n/2
    const unsigned int arrow_id = get_global_id(0);

    if (arrow_id >= n/2) {
        return;
    }

    const unsigned int arrows_in_strip = local_strip / 2;
    const unsigned int arrow_group_id = arrow_id / arrows_in_strip;
    const unsigned int arrow_group_offset = arrow_id % arrows_in_strip;
    const unsigned int left_idx = arrow_group_id * local_strip + arrow_group_offset;
    const unsigned int right_idx = left_idx + arrows_in_strip;

    const float is_blue = (left_idx & global_strip) == 0 ? 1.f : 0.f;

    const float left = as[left_idx];
    const float right = as[right_idx];

    as[left_idx] = is_blue * min(left, right) + (1.f - is_blue) * max(left, right);
    as[right_idx] = is_blue * max(left, right) + (1.f - is_blue) * min(left, right);
}

