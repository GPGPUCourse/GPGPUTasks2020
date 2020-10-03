#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/matrix_transpose_cl.h"

#include <vector>
#include <iostream>
#include <stdexcept>


void run(gpu::Device& device, unsigned int tile_size, bool fix_bank) {
    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    unsigned int M = 1024;
    unsigned int K = 1024;

    std::vector<float> as(M*K, 0);
    std::vector<float> as_t(M*K, 0);

    FastRandom r(M+K);
    for (unsigned int i = 0; i < as.size(); ++i) {
        as[i] = r.nextf();
    }
    std::cout << "Data generated for M=" << M << ", K=" << K << "!" << std::endl;

    gpu::gpu_mem_32f as_gpu, as_t_gpu;
    as_gpu.resizeN(M*K);
    as_t_gpu.resizeN(K*M);

    as_gpu.writeN(as.data(), M*K);

    std::string defines = "-D FIX_BANK=" + std::to_string(fix_bank ? 1 : 0) + " -D TILE_SIZE=" + std::to_string(tile_size);

    ocl::Kernel matrix_transpose_kernel(matrix_transpose, matrix_transpose_length, "matrix_transpose", defines);
    matrix_transpose_kernel.compile();

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int x_work_size = K;
            unsigned int y_work_size = M;

            matrix_transpose_kernel.exec(
                    gpu::WorkSize(tile_size, tile_size, x_work_size, y_work_size),
                    as_gpu, as_t_gpu, K, M);

            t.nextLap();
        }
        std::cout << "Params: fix_bank=" + to_string(fix_bank) + ", tile size=" + to_string(tile_size) << std::endl;
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << M*K/1000.0/1000.0 / t.lapAvg() << " millions/s" << std::endl;
        std::cout << std::endl;
    }

    as_t_gpu.readN(as_t.data(), M*K);

    // Проверяем корректность результатов
    for (int j = 0; j < M; ++j) {
        for (int i = 0; i < K; ++i) {
            float a = as[j * K + i];
            float b = as_t[i * M + j];
            if (a != b) {
                std::string err = "Not the same! Params: fix_bank=" + to_string(fix_bank) + ", tile size=" + to_string(tile_size);
                throw std::runtime_error(err);
            }
        }
    }
}

int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    try {
        run(device, 8, false);
        run(device, 8, true);
        run(device, 16, false);
        run(device, 16, true);
    } catch (const std::runtime_error& e) {
        std::cerr << e.what();
        return 1;
    }

    return 0;
}
