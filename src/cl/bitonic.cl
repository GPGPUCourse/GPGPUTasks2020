#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORKGROUP_SIZE 128
typedef unsigned int uint;

//  b = 00011111
// id = abcdefgh
//           ^
//           s (= 2)
// id_to_start_idx(id)
//    = 000ef0gh
// id - number to shift
// b - max value of id
// s - pos where shift starts
// expected (1<<s) <= b
uint id_to_start_idx(uint id, uint b, uint s) {
    id &= b;
    return ((id >>s <<s+1)|(id&((1<<s)-1)))&b;
}

__kernel void bitonic_step(__global float* as, uint block_size, uint swap_dist_log)
{
    const uint id = get_global_id(0);

    uint i;
    uint j;
    float a_i;
    float a_j;

    #define SWAP4(start, shift_swap, shift_start_reverse) {\
        i = start;\
        j = i + shift_swap;\
        a_i = as[i];\
        a_j = as[j];\
        as[i] = a_i <= a_j ? a_i : a_j;\
        as[j] = a_i <= a_j ? a_j : a_i;\
        i = i + shift_start_reverse;\
        j = i + shift_swap;\
        a_i = as[i];\
        a_j = as[j];\
        as[i] = a_i >= a_j ? a_i : a_j;\
        as[j] = a_i >= a_j ? a_j : a_i;\
    }

    SWAP4(id/(block_size/2)*block_size*2 + id_to_start_idx(id, block_size-1, swap_dist_log), (1<<swap_dist_log), block_size);
}

__kernel void bitonic_finisher(__global float* as, uint block_size /*length of as*/, uint swap_dist_log)
{
    // TODO
    const uint id = get_global_id(0);

    uint i;
    uint j;
    float a_i;
    float a_j;

    #define SWAP2(start, shift_swap) {\
        i = start;\
        j = i + shift_swap;\
        a_i = as[i];\
        a_j = as[j];\
        as[i] = a_i <= a_j ? a_i : a_j;\
        as[j] = a_i <= a_j ? a_j : a_i;\
    }

    SWAP2(id_to_start_idx(id, block_size-1, swap_dist_log), (1<<swap_dist_log));
}

__kernel void bitonic_begin512(__global float* as)
{
    const uint id = get_local_id(0);
    const uint offset = get_global_id(0)/WORKGROUP_SIZE*(4*WORKGROUP_SIZE);

// если тут поставить один, то массив корраптится почему-то..........
// и работает медленнее
#if 0
    #define USE_LOCAL_MEMORY
    __local float array[4*WORKGROUP_SIZE];
    array[id] = as[offset + id];
    array[WORKGROUP_SIZE + id] = as[offset + WORKGROUP_SIZE + id];
    array[2*WORKGROUP_SIZE + id] = as[offset + 2*WORKGROUP_SIZE + id];
    array[3*WORKGROUP_SIZE + id] = as[offset + 3*WORKGROUP_SIZE + id];
#endif

    uint i;
    uint j;
    float a_i;
    float a_j;

#ifdef USE_LOCAL_MEMORY
    #define SWAP4INLOCAL(start, shift_swap, shift_start_reverse) {\
        barrier(CLK_LOCAL_MEM_FENCE);\
        i = start;\
        j = i + shift_swap;\
        a_i = array[i];\
        a_j = array[j];\
        array[i] = a_i <= a_j ? a_i : a_j;\
        array[j] = a_i <= a_j ? a_j : a_i;\
        i = i + shift_start_reverse;\
        j = i + shift_swap;\
        a_i = array[i];\
        a_j = array[j];\
        array[i] = a_i >= a_j ? a_i : a_j;\
        array[j] = a_i >= a_j ? a_j : a_i;\
    }
    #define SWAP4 SWAP4INLOCAL
#else
    #define SWAP4(start, shift_swap, shift_start_reverse) {\
        barrier(CLK_LOCAL_MEM_FENCE);\
        i = offset + start;\
        j = i + shift_swap;\
        a_i = as[i];\
        a_j = as[j];\
        as[i] = a_i <= a_j ? a_i : a_j;\
        as[j] = a_i <= a_j ? a_j : a_i;\
        i = i + shift_start_reverse;\
        j = i + shift_swap;\
        a_i = as[i];\
        a_j = as[j];\
        as[i] = a_i >= a_j ? a_i : a_j;\
        as[j] = a_i >= a_j ? a_j : a_i;\
    }
#endif

    // 2
    SWAP4(id/1*4 + id_to_start_idx(id, 2-1, 0), 1, 2);

    // 4
    SWAP4(id/2*8 + id_to_start_idx(id, 4-1, 1), 2, 4);
    SWAP4(id/2*8 + id_to_start_idx(id, 4-1, 0), 1, 4);

    // 8
    SWAP4(id/4*16 + id_to_start_idx(id, 8-1, 2), 4, 8);
    SWAP4(id/4*16 + id_to_start_idx(id, 8-1, 1), 2, 8);
    SWAP4(id/4*16 + id_to_start_idx(id, 8-1, 0), 1, 8);

    // 16
    SWAP4(id/8*32 + id_to_start_idx(id, 16-1, 3), 8, 16);
    SWAP4(id/8*32 + id_to_start_idx(id, 16-1, 2), 4, 16);
    SWAP4(id/8*32 + id_to_start_idx(id, 16-1, 1), 2, 16);
    SWAP4(id/8*32 + id_to_start_idx(id, 16-1, 0), 1, 16);

    // 32
    SWAP4(id/16*64 + id_to_start_idx(id, 32-1, 4), 16, 32);
    SWAP4(id/16*64 + id_to_start_idx(id, 32-1, 3), 8, 32);
    SWAP4(id/16*64 + id_to_start_idx(id, 32-1, 2), 4, 32);
    SWAP4(id/16*64 + id_to_start_idx(id, 32-1, 1), 2, 32);
    SWAP4(id/16*64 + id_to_start_idx(id, 32-1, 0), 1, 32);

    // 64
    //printf("\n[%d, %d, %d]\n", id/32*128 + id_to_start_idx(id, 64-1, 5) + 32 + 64, id, id_to_start_idx(id, 64-1, 5));
    SWAP4(id/32*128 + id_to_start_idx(id, 64-1, 5), 32, 64);
    SWAP4(id/32*128 + id_to_start_idx(id, 64-1, 4), 16, 64);
    SWAP4(id/32*128 + id_to_start_idx(id, 64-1, 3), 8, 64);
    SWAP4(id/32*128 + id_to_start_idx(id, 64-1, 2), 4, 64);
    SWAP4(id/32*128 + id_to_start_idx(id, 64-1, 1), 2, 64);
    SWAP4(id/32*128 + id_to_start_idx(id, 64-1, 0), 1, 64);

    // 128
    SWAP4(id/64*256 + id_to_start_idx(id, 128-1, 6), 64, 128);
    SWAP4(id/64*256 + id_to_start_idx(id, 128-1, 5), 32, 128);
    SWAP4(id/64*256 + id_to_start_idx(id, 128-1, 4), 16, 128);
    SWAP4(id/64*256 + id_to_start_idx(id, 128-1, 3), 8, 128);
    SWAP4(id/64*256 + id_to_start_idx(id, 128-1, 2), 4, 128);
    SWAP4(id/64*256 + id_to_start_idx(id, 128-1, 1), 2, 128);
    SWAP4(id/64*256 + id_to_start_idx(id, 128-1, 0), 1, 128);

    // 256
    SWAP4(id/128*512 + id_to_start_idx(id, 256-1, 7), 128, 256);
    SWAP4(id/128*512 + id_to_start_idx(id, 256-1, 6), 64, 256);
    SWAP4(id/128*512 + id_to_start_idx(id, 256-1, 5), 32, 256);
    SWAP4(id/128*512 + id_to_start_idx(id, 256-1, 4), 16, 256);
    SWAP4(id/128*512 + id_to_start_idx(id, 256-1, 3), 8, 256);
    SWAP4(id/128*512 + id_to_start_idx(id, 256-1, 2), 4, 256);
    SWAP4(id/128*512 + id_to_start_idx(id, 256-1, 1), 2, 256);
    SWAP4(id/128*512 + id_to_start_idx(id, 256-1, 0), 1, 256);

#ifdef USE_LOCAL_MEMORY
    as[offset + id] = array[id];
    as[offset + WORKGROUP_SIZE + id] = array[WORKGROUP_SIZE + id];
    as[offset + 2*WORKGROUP_SIZE + id] = array[2*WORKGROUP_SIZE + id];
    as[offset + 3*WORKGROUP_SIZE + id] = array[3*WORKGROUP_SIZE + id];
#endif
}


#if 0
// use blue block with size 512 on [as_shifted, as_shifted + 512)
void bitonic_blue512_reuse(__global float* as_shifted)
{
    // TODO
    const uint id = get_local_id(0);
    __local float array[4*WORKGROUP_SIZE];
    array[id] = as_shifted[id];
    array[WORKGROUP_SIZE + id] = as_shifted[WORKGROUP_SIZE + id];
    array[2*WORKGROUP_SIZE + id] = as_shifted[2*WORKGROUP_SIZE + id];
    array[3*WORKGROUP_SIZE + id] = as_shifted[3*WORKGROUP_SIZE + id];

    uint i;
    uint j;
    float a_i;
    float a_j;
    float cond;

    #define SWAP2BARRIER(start, shift_swap) {\
        barrier(CLK_LOCAL_MEM_FENCE);\
        i = start;\
        j = i + shift_swap;\
        a_i = array[i];\
        a_j = array[j];\
        cond = a_i <= a_j ? 1.0f : 0.0f;\
        array[i] = cond * a_i + (1.0f - cond) * a_j;\
        array[j] = (1.0f - cond) * a_i + cond * a_j;\
    }

    #define SWAP2(start, shift_swap) {\
        i = start;\
        j = i + shift_swap;\
        a_i = array[i];\
        a_j = array[j];\
        cond = a_i <= a_j ? 1.0f : 0.0f;\
        array[i] = cond * a_i + (1.0f - cond) * a_j;\
        array[j] = (1.0f - cond) * a_i + cond * a_j;\
    }

    // 512
    SWAP2BARRIER(id_to_start_idx(id, 512-1, 8), 256);
    SWAP2(id_to_start_idx(id+128, 512-1, 8), 256);
    SWAP2BARRIER(id_to_start_idx(id, 512-1, 7), 128);
    SWAP2(id_to_start_idx(id+128, 512-1, 7), 128);
    SWAP2BARRIER(id_to_start_idx(id, 512-1, 6), 64);
    SWAP2(id_to_start_idx(id+128, 512-1, 6), 64);
    SWAP2BARRIER(id_to_start_idx(id, 512-1, 5), 32);
    SWAP2(id_to_start_idx(id+128, 512-1, 5), 32);
    SWAP2BARRIER(id_to_start_idx(id, 512-1, 4), 16);
    SWAP2(id_to_start_idx(id+128, 512-1, 4), 16);
    SWAP2BARRIER(id_to_start_idx(id, 512-1, 3), 8);
    SWAP2(id_to_start_idx(id+128, 512-1, 3), 8);
    SWAP2BARRIER(id_to_start_idx(id, 512-1, 2), 4);
    SWAP2(id_to_start_idx(id+128, 512-1, 2), 4);
    SWAP2BARRIER(id_to_start_idx(id, 512-1, 1), 2);
    SWAP2(id_to_start_idx(id+128, 512-1, 1), 2);
    SWAP2BARRIER(id_to_start_idx(id, 512-1, 0), 1);
    SWAP2(id_to_start_idx(id+128, 512-1, 0), 1);

    barrier(CLK_LOCAL_MEM_FENCE);

    as_shifted[id] = array[id];
    as_shifted[WORKGROUP_SIZE + id] = array[WORKGROUP_SIZE + id];
    as_shifted[2*WORKGROUP_SIZE + id] = array[2*WORKGROUP_SIZE + id];
    as_shifted[3*WORKGROUP_SIZE + id] = array[3*WORKGROUP_SIZE + id];
}

// use green block with size 512 on [as_shifted, as_shifted + 512)
void bitonic_green512_reuse(__global float* as_shifted)
{
    // TODO
    const uint id = get_local_id(0);
    __local float array[4*WORKGROUP_SIZE];
    array[id] = as_shifted[id];
    array[WORKGROUP_SIZE + id] = as_shifted[WORKGROUP_SIZE + id];
    array[2*WORKGROUP_SIZE + id] = as_shifted[2*WORKGROUP_SIZE + id];
    array[3*WORKGROUP_SIZE + id] = as_shifted[3*WORKGROUP_SIZE + id];

    uint i;
    uint j;
    float a_i;
    float a_j;
    float cond;

    #define SWAP2BACKBARRIER(start, shift_swap) {\
        barrier(CLK_LOCAL_MEM_FENCE);\
        i = start;\
        j = i + shift_swap;\
        a_i = array[i];\
        a_j = array[j];\
        cond = a_i >= a_j ? 1.0f : 0.0f;\
        array[i] = cond * a_i + (1.0f - cond) * a_j;\
        array[j] = (1.0f - cond) * a_i + cond * a_j;\
    }

    #define SWAP2BACK(start, shift_swap) {\
        i = start;\
        j = i + shift_swap;\
        a_i = array[i];\
        a_j = array[j];\
        cond = a_i >= a_j ? 1.0f : 0.0f;\
        array[i] = cond * a_i + (1.0f - cond) * a_j;\
        array[j] = (1.0f - cond) * a_i + cond * a_j;\
    }

    // 512
    SWAP2BACKBARRIER(id_to_start_idx(id, 512-1, 8), 256);
    SWAP2BACK(id_to_start_idx(id+128, 512-1, 8), 256);
    SWAP2BACKBARRIER(id_to_start_idx(id, 512-1, 7), 128);
    SWAP2BACK(id_to_start_idx(id+128, 512-1, 7), 128);
    SWAP2BACKBARRIER(id_to_start_idx(id, 512-1, 6), 64);
    SWAP2BACK(id_to_start_idx(id+128, 512-1, 6), 64);
    SWAP2BACKBARRIER(id_to_start_idx(id, 512-1, 5), 32);
    SWAP2BACK(id_to_start_idx(id+128, 512-1, 5), 32);
    SWAP2BACKBARRIER(id_to_start_idx(id, 512-1, 4), 16);
    SWAP2BACK(id_to_start_idx(id+128, 512-1, 4), 16);
    SWAP2BACKBARRIER(id_to_start_idx(id, 512-1, 3), 8);
    SWAP2BACK(id_to_start_idx(id+128, 512-1, 3), 8);
    SWAP2BACKBARRIER(id_to_start_idx(id, 512-1, 2), 4);
    SWAP2BACK(id_to_start_idx(id+128, 512-1, 2), 4);
    SWAP2BACKBARRIER(id_to_start_idx(id, 512-1, 1), 2);
    SWAP2BACK(id_to_start_idx(id+128, 512-1, 1), 2);
    SWAP2BACKBARRIER(id_to_start_idx(id, 512-1, 0), 1);
    SWAP2BACK(id_to_start_idx(id+128, 512-1, 0), 1);

    barrier(CLK_LOCAL_MEM_FENCE);

    as_shifted[id] = array[id];
    as_shifted[WORKGROUP_SIZE + id] = array[WORKGROUP_SIZE + id];
    as_shifted[2*WORKGROUP_SIZE + id] = array[2*WORKGROUP_SIZE + id];
    as_shifted[3*WORKGROUP_SIZE + id] = array[3*WORKGROUP_SIZE + id];
}
#endif
