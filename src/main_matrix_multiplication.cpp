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

    int benchmarkingIters = 1; // TODO пока тестируетесь удобно выставить единицу
    unsigned int H1 = 1024;
    unsigned int K = 1024;
    unsigned int W1 = K;
    unsigned int H2 = K;
    unsigned int W2 = 1024;
    unsigned int HR = H1;
    unsigned int WR = W2;
    const size_t gflops = ((size_t) H1 * K * W2 * 2) / (1000 * 1000 * 1000); // умножить на два, т.к. операция сложения и умножения

    std::vector<float> as(H1*W1, 0);
    std::vector<float> bs(H2*W2, 0);
    std::vector<float> cs(HR*WR, 0);

    FastRandom r(H1+K+W2);
    for (unsigned int i = 0; i < as.size(); ++i) {
        as[i] = r.nextf();
    }
    for (unsigned int i = 0; i < bs.size(); ++i) {
        bs[i] = r.nextf();
    }
    std::cout << "Data generated for H1=" << H1 << ", K=" << K << ", W2=" << W2 << "!" << std::endl;

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            for (int j = 0; j < H1; ++j) {
                for (int i = 0; i < W2; ++i) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; ++k) {
                        sum += as.data()[j * K + k] * bs.data()[k * W2 + i];
                    }
                    cs.data()[j * W2 + i] = sum;
                }
            }
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << gflops / t.lapAvg() << " GFlops" << std::endl;
    }

    const std::vector<float> cs_cpu_reference = cs;

    gpu::gpu_mem_32f as_gpu, bs_gpu, cs_gpu;
    as_gpu.resizeN(H1*W1);
    bs_gpu.resizeN(H2*W2);
    cs_gpu.resizeN(HR*WR);

    as_gpu.writeN(as.data(), H1*W1);
    bs_gpu.writeN(bs.data(), H2*W2);

    ocl::Kernel matrix_multiplication_kernel(matrix_multiplication, matrix_multiplication_length, "matrix_multiplication2");
    matrix_multiplication_kernel.compile();

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            // TODO
            //unsigned int work_group_size = 8;

            unsigned int work_group_size = 8 * 4;

            unsigned int global_work_size_x = work_group_size / 4 * ((WR + work_group_size - 1) / work_group_size);
            unsigned int global_work_size_y = work_group_size / 4 * ((HR + work_group_size - 1) / work_group_size);

            work_group_size = 8;

            matrix_multiplication_kernel.exec(gpu::WorkSize(work_group_size, work_group_size,
              global_work_size_x, global_work_size_y), as_gpu, bs_gpu, cs_gpu, H1, K, W2);

            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << gflops / t.lapAvg() << " GFlops" << std::endl;
    }

    cs_gpu.readN(cs.data(), HR*WR);

    // Проверяем корректность результатов
    double diff_sum = 0;
    for (int i = 0; i < HR * WR; ++i) {
        double a = cs[i];
        double b = cs_cpu_reference[i];
        if (a != 0.0 && b != 0.0) {
            double diff = fabs(a - b) / std::max(fabs(a), fabs(b));
            diff_sum += diff;
        }
    }

    double diff_avg = diff_sum / (HR * WR);
    std::cout << "Average difference: " << diff_avg * 100.0 << "%" << std::endl;
    if (diff_avg > 0.01) {
        std::cerr << "Too big difference!" << std::endl;
        return 1;
    }

    return 0;
}
