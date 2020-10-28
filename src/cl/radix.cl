#define WORKGROUP_SIZE 256

__kernel void radix8(__global const unsigned int *as, __global unsigned int *bs, __global const unsigned int *rs8,
                    unsigned int n, unsigned int n_ext, unsigned int shift) {
    unsigned int g = get_global_id(0);
    unsigned int l = get_local_id(0);
    __local unsigned int al[WORKGROUP_SIZE];
    __local unsigned int ps[8][WORKGROUP_SIZE];
    __local unsigned int ys[8][WORKGROUP_SIZE];

    unsigned int asg, asg7;
    int bl = 0;
    if (g < n) {
        asg = as[g];
        asg7 = asg >> shift & 7;
        for (int t = 0; t < 8; t++) {
            ps[t][l] = asg7 == t;
            ys[t][l] = 0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int i = l + 1;

    for (int t = 0; t < 8; t++) {
        for (int pow_2 = 2; pow_2 <= 256; pow_2 *= 2) {
            if (i % pow_2 == 0 && g < n) {
                ps[t][l] += ps[t][l - pow_2 / 2];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        for (int pow_2 = 1; pow_2 <= 256; pow_2 *= 2) {
            if (i & pow_2 && g < n) {
                ys[t][l] += ps[t][i / pow_2 * pow_2 - 1];
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (g < n) {
        unsigned int offsets = 0;

        // считаем меньшие цифры во всех группах
        for (int t = 0; t < asg7; t++) {
            offsets += rs8[n_ext - 1 + t * n_ext];
        }

        // считаем эту цифру в предыдущих группах
        if (g / WORKGROUP_SIZE >= 1) {
            offsets += rs8[g / WORKGROUP_SIZE - 1 + asg7 * n_ext];
        }

        // считаем эту цифру в этой группе
        if (l > 0) {
            offsets += ys[asg7][l - 1];
        }
        bs[offsets] = asg;
    }
}

// по исходному массиву считаем локальные суммы в группах по 256 элементов и кладём их на 1 уровень
__kernel void partial_sum_main8(__global const unsigned int *as, __global unsigned int *rs8,
                                unsigned int n, unsigned int n_ext, unsigned int shift) {
    unsigned int g = get_global_id(0);
    unsigned int l = get_local_id(0);

    // ps - суммы по последним k элементам, где k - младшая степень 2 для текущего элемента
    // ys - частичные суммы для элементов воркгруппы
    __local unsigned int ps[8][WORKGROUP_SIZE];
    __local unsigned int ys[8][WORKGROUP_SIZE];

    // инициализируем ps 1 и 0
    if (g < n) {
        unsigned int asg = as[g];
        for (int t = 0; t < 8; t++) {
            ps[t][l] = (asg >> shift & 7) == t;
            ys[t][l] = 0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int i = l + 1;

    for (int t = 0; t < 8; t++) {
        for (int pow_2 = 2; pow_2 <= 256; pow_2 *= 2) {
            if (i % pow_2 == 0 && g < n) {
                ps[t][l] += ps[t][l - pow_2 / 2];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        for (int pow_2 = 1; pow_2 <= 256; pow_2 *= 2) {
            if (i & pow_2 && g < n) {
                ys[t][l] += ps[t][i / pow_2 * pow_2 - 1];
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (g < n) {
        if (l == WORKGROUP_SIZE - 1 || g == n - 1) {
            for (int t = 0; t < 8; t++) {
                rs8[g / WORKGROUP_SIZE + t * n_ext] = ys[t][l];
            }
        }
    }
}

// считаем префиксные суммы для внутренних узлов
__kernel void partial_sum8(__global unsigned int *rs8, unsigned int ni, int offset, unsigned int n_ext) {
    unsigned int g = get_global_id(0);
    unsigned int l = get_local_id(0);
    __local unsigned int ps[8][WORKGROUP_SIZE];
    __local unsigned int ys[8][WORKGROUP_SIZE];

    if (g < ni) {
        for (int t = 0; t < 8; t++) {
            ps[t][l] = rs8[offset + g + t * n_ext];
            ys[t][l] = 0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int i = l + 1;

    for (int t = 0; t < 8; t++) {
        for (int pow_2 = 2; pow_2 <= 256; pow_2 *= 2) {
            if (i % pow_2 == 0 && g < ni) {
                ps[t][l] += ps[t][l - pow_2 / 2];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        for (int pow_2 = 1; pow_2 <= 256; pow_2 *= 2) {
            if (i & pow_2 && g < ni) {
                ys[t][l] += ps[t][i / pow_2 * pow_2 - 1];
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int t = 0; t < 8; t++) {
        if (g < ni) {
            rs8[offset + g + t * n_ext] = ys[t][l];
            if (l == WORKGROUP_SIZE - 1 || g == ni - 1) {
                rs8[offset + g / WORKGROUP_SIZE + t * n_ext + ni] = ys[t][l];
            }
        }
    }
}

// собираем префиксные суммы на 0 уровень
__kernel void partial_sum_gather8(__global unsigned int *rs, __global unsigned int *ns,
                                  unsigned int ns_size, unsigned int n_ext) {

    for (int t = 0; t < 8; t++) {
        unsigned int g = get_global_id(0);

        if (g < ns[1]) {
            unsigned int s = rs[g + t * n_ext];

            for (int i = 1, offset = ns[1]; g > 1 && i < ns_size; offset += ns[++i]) {
                if ((g / WORKGROUP_SIZE) % WORKGROUP_SIZE != 0) {
                    s += rs[offset + g / WORKGROUP_SIZE - 1 + t * n_ext];
                }
                g /= WORKGROUP_SIZE;
            }

            rs[get_global_id(0) + t * n_ext] = s;
        }
    }
}