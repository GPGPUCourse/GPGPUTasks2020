#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/matrix_multiplication_cl.h"

#include <vector>
#include <iostream>
#include <stdexcept>


int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10; // TODO пока тестируетесь удобно выставить единицу
    unsigned int M = 1024;
    unsigned int K = 1024;
    unsigned int N = 1024;

    // unsigned int M = 128;
    // unsigned int K = 128;
    // unsigned int N = 128;
    const size_t gflops = ((size_t) M * K * N * 2) / (1000 * 1000 * 1000); // умножить на два, т.к. операция сложения и умножения

    std::vector<float> as(M*K, 0);
    std::vector<float> bs(K*N, 0);
    std::vector<float> cs(M*N, 0);

    FastRandom r(M+K+N);
    for (unsigned int i = 0; i < as.size(); ++i) {
        as[i] = r.nextf();
    }
    for (unsigned int i = 0; i < bs.size(); ++i) {
        bs[i] = r.nextf();
    }
    std::cout << "Data generated for M=" << M << ", K=" << K << ", N=" << N << "!" << std::endl;

    {
        timer t;
        std::cout.precision(2);
        std::cout << std::fixed;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            for (int j = 0; j < M; ++j) {
                for (int i = 0; i < N; ++i) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; ++k) {
                        sum += as.data()[j * K + k] * bs.data()[k * N + i];
                        // std::cout << "from: " << j * K + k << ' ' << k * N + i << " to: " << j * N + i << " m1 = " << as[j * K + k] << " m2 = " << bs[k * N + i] << " val = " << as[j * K + k] * bs[k * N + i] << std::endl;
                    }
                    cs.data()[j * N + i] = sum;
                }
            }
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << gflops / t.lapAvg() << " GFlops" << std::endl;
    }

    std::cout << "===============================================" << std::endl;

    const std::vector<float> cs_cpu_reference = cs;

    gpu::gpu_mem_32f as_gpu, bs_gpu, cs_gpu;
    as_gpu.resizeN(M*K);
    bs_gpu.resizeN(K*N);
    cs_gpu.resizeN(M*N);

    std::fill(cs.begin(), cs.end(), 0);
    as_gpu.writeN(as.data(), M*K);
    bs_gpu.writeN(bs.data(), K*N);
    cs_gpu.writeN(cs.data(), M*N);

    ocl::Kernel matrix_multiplication_kernel(matrix_multiplication, matrix_multiplication_length, "matrix_multiplication");
    matrix_multiplication_kernel.compile();

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int square_size = 32;
            unsigned int global_work_size_m = gpu::divup(M, square_size) * square_size;
            unsigned int global_work_size_n = gpu::divup(N, square_size) * square_size;
            matrix_multiplication_kernel.exec(
                gpu::WorkSize(
                    square_size, square_size,
                    global_work_size_m, global_work_size_n
                ), 
                as_gpu, bs_gpu, cs_gpu, M, K, N);
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << gflops / t.lapAvg() << " GFlops" << std::endl;
    }

    cs_gpu.readN(cs.data(), M*N);

    // for (int i = 0; i < M * N; i++)
    //     std::cout << cs[i] << ' ';
    // std::cout << std::endl;

    // for (int i = 0; i < M * N; i++)
    //     std::cout << cs_cpu_reference[i] << ' ';
    // std::cout << std::endl;


    // Проверяем корректность результатов
    double diff_sum = 0;
    for (int i = 0; i < M * N; ++i) {
        double a = cs[i];
        double b = cs_cpu_reference[i];
        if (a != 0.0 && b != 0.0) {
            double diff = fabs(a - b) / std::max(fabs(a), fabs(b));
            diff_sum += diff;
        }
    }

    double diff_avg = diff_sum / (M * N);
    std::cout << "Average difference: " << diff_avg * 100.0 << "%" << std::endl;
    if (diff_avg > 0.01) {
        std::cerr << "Too big difference!" << std::endl;
        return 1;
    }

    return 0;
}
