#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/matrix_multiplication_cl.h"

#include <vector>
#include <iostream>
#include <stdexcept>

#define LOCAL_WH 16

void printMatrix(const float * data, unsigned int I, unsigned int J) {
    for (unsigned int idx = 0; idx < I*J; ++idx) {
       std::cout << data[idx] << ((idx+1)%J==0 ? "\n" : "\t");
    }
}

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
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            for (int m = 0; m < M; ++m) {
                for (int n = 0; n < N; ++n) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; ++k) {
                        sum += as.data()[m * K + k] * bs.data()[k * N + n];
                    }
                    cs.data()[m * N + n] = sum;
                }
            }
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << gflops / t.lapAvg() << " GFlops" << std::endl;
    }

    const std::vector<float> cs_cpu_reference = cs;

    gpu::gpu_mem_32f as_gpu, bs_gpu, cs_gpu;
    as_gpu.resizeN(M*K);
    bs_gpu.resizeN(K*N);
    cs_gpu.resizeN(M*N);

    as_gpu.writeN(as.data(), M*K);
    bs_gpu.writeN(bs.data(), K*N);

    std::string defines = "-D LOCAL_WH=" + std::to_string(LOCAL_WH);
    ocl::Kernel matrix_multiplication_kernel(matrix_multiplication, matrix_multiplication_length, "matrix_multiplication", defines);
    matrix_multiplication_kernel.compile(true);

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            // TODO
            matrix_multiplication_kernel.exec(gpu::WorkSize(LOCAL_WH, LOCAL_WH, N, M), as_gpu, bs_gpu, cs_gpu, M, K, N);

            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << gflops / t.lapAvg() << " GFlops" << std::endl;
    }

    cs_gpu.readN(cs.data(), M*N);

    // Проверяем корректность результатов
    double diff_sum = 0;
    for (int i = 0; i < M * N; ++i) {
        double a = cs[i];
        double b = cs_cpu_reference[i];
        if (a != 0.0 || b != 0.0) {
            double diff = fabs(a - b) / std::max(fabs(a), fabs(b));
            diff_sum += diff;
        }
        //std::cout << (fabs(a - b) < 0.000000001) << ((i+1)%M == 0 ? "\n" : " ");
    }

    double diff_avg = diff_sum / (M * N);
    std::cout << "Average difference: " << diff_avg * 100.0 << "%" << std::endl;
    if (diff_avg > 0.01) {
        std::cerr << "Too big difference!" << std::endl;
        return 1;
    }

    return 0;
}
