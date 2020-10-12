#define TILE_SIZE 16

__kernel void matrix_multiplication(__global const float* as,
                                    __global const float* bs,
                                    __global       float* cs,
                                    unsigned int M,
                                    unsigned int K,
                                    unsigned int N)
{
    __local float A_k[TILE_SIZE][TILE_SIZE+1]; // +1 shift to resolve bank conflict
    __local float B_k[TILE_SIZE][TILE_SIZE+1];

    int glob_i = get_global_id(0);
    int glob_j = get_global_id(1);
    int i = get_local_id(0);
    int j = get_local_id(1);

    float C_k = 0;
    for (int k = 0; k < K / TILE_SIZE; ++k)
    {
        // load part of the line and part of the row
        A_k[j][i] = as[glob_j * K + k * TILE_SIZE + i];
        B_k[j][i] = bs[(j + k * TILE_SIZE) * N + glob_i];
        // barrier
        barrier(CLK_LOCAL_MEM_FENCE);
        // accumulate
        for (int idx = 0; idx < TILE_SIZE; ++idx)
        {
            C_k += A_k[j][idx] * B_k[idx][i];
        }
        // make love not WAR
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // write back to vram
    cs[glob_j * N + glob_i] = C_k;
} 