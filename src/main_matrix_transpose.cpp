#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/matrix_transpose_cl.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <stdexcept>


int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    unsigned int M = 4567;
    unsigned int K = 8765;

    std::vector<float> as(M*K, 0);
    std::vector<float> as_t(M*K, 0);

    FastRandom r(M+K);
    for (unsigned int i = 0; i < as.size(); ++i) {
        // as[i] = (i / K) + (i % K) * .001;
        as[i] = r.nextf();
    }
    std::cout << "Data generated for M=" << M << ", K=" << K << "!" << std::endl;

    gpu::gpu_mem_32f as_gpu, as_t_gpu;
    as_gpu.resizeN(M*K);
    as_t_gpu.resizeN(K*M);

    as_gpu.writeN(as.data(), M*K);

    ocl::Kernel matrix_transpose_kernel(matrix_transpose, matrix_transpose_length, "matrix_transpose");
    matrix_transpose_kernel.compile();

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            // TODO
            unsigned int S = 16;
            unsigned int work_group_size_X = S;
            unsigned int work_group_size_Y = S;
            unsigned int global_work_size_X = (K + work_group_size_X - 1) / work_group_size_X * work_group_size_X;
            unsigned int global_work_size_Y = (M + work_group_size_Y - 1) / work_group_size_Y * work_group_size_Y;
            matrix_transpose_kernel.exec(gpu::WorkSize(work_group_size_X, work_group_size_Y, global_work_size_X, global_work_size_Y), as_gpu, as_t_gpu, M, K);

            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << M*K/1000.0/1000.0 / t.lapAvg() << " millions/s" << std::endl;
    }

    as_t_gpu.readN(as_t.data(), M*K);
    /*
    // Выводим результаты
    std::ofstream fout("transpose_res.txt");
    for (int j = 0; j < M; ++j)
        for (int i = 0; i < K; ++i)
            fout << as[j * K + i] << " \n"[i == K - 1];
    fout << "===================================================================\n";
    for (int j = 0; j < M; ++j)
        for (int i = 0; i < K; ++i)
            fout << as_t[j * K + i] << " \n"[i == K - 1];
*/
    // Проверяем корректность результатов
    for (int j = 0; j < M; ++j) {
        for (int i = 0; i < K; ++i) {
            float a = as[j * K + i];
            float b = as_t[i * M + j];
            if (a != b) {
                std::cerr << "Not the same!" << std::endl;
                return 1;
            }
        }
    }

    return 0;
}
