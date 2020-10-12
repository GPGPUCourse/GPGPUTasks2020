#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/device.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/opencl/device_info.h>

#include "cl/matrix_transpose_cl.h"
#include "cl/matrix_multiplication_cl.h"

#include <cassert>
#include <vector>
#include <iostream>
#include <stdexcept>

#define WITH_CPU

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

#ifdef WITH_CPU
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            for (int j = 0; j < M; ++j) {
                for (int i = 0; i < N; ++i) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; ++k) {
                        sum += as.data()[j * K + k] * bs.data()[k * N + i];
                    }
                    cs.data()[j * N + i] = sum;
                }
            }
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << gflops / t.lapAvg() << " GFlops" << std::endl;
    }
#endif

    const std::vector<float> cs_cpu_reference = cs;

    gpu::gpu_mem_32f as_gpu, bs_gpu, bs_t_gpu, cs_gpu;
    as_gpu.resizeN(M*K);
    bs_gpu.resizeN(K*N);
    bs_t_gpu.resizeN(N*K);
    cs_gpu.resizeN(M*N);

    as_gpu.writeN(as.data(), M*K);
    bs_gpu.writeN(bs.data(), K*N);

    ocl::Kernel matrix_transpose_kernel(matrix_transpose, matrix_transpose_length, "matrix_transpose");
    matrix_transpose_kernel.compile();

    ocl::Kernel matrix_multiplication_kernel(matrix_multiplication, matrix_multiplication_length, "matrix_multiplication");
    matrix_multiplication_kernel.compile();

    {
        ocl::DeviceInfo deviceInfo;
        deviceInfo.init(device.device_id_opencl);
        
        const size_t warp_size = deviceInfo.warp_size != 0 
            ? deviceInfo.warp_size 
            : (deviceInfo.wavefront_width != 0 
                ? deviceInfo.wavefront_width 
                : 1 // fallback for GPU-less machines (like our CI)
            );

        size_t work_group_side = 1;
        while ((work_group_side * work_group_side << 2) <= std::min(deviceInfo.max_workgroup_size, warp_size * warp_size)) {
            work_group_side <<= 1;
        }

        // we want tiles of WARP_SIZE x WARP_SIZE which will be filled line by line 
        // by whole warps and then subdivided into smaller WG_SIDE x WG_SIDE for transposition
        assert(work_group_side * work_group_side % warp_size == 0 && "wrong assumptions");

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            {
                const size_t global_tiles_x = (K + warp_size - 1) / warp_size;
                const size_t global_tiles_y = (N + warp_size - 1) / warp_size;

                const size_t global_work_size_x = global_tiles_x * work_group_side;
                const size_t global_work_size_y = global_tiles_y * work_group_side;

                matrix_transpose_kernel.exec(
                    gpu::WorkSize(work_group_side, work_group_side, global_work_size_x, global_work_size_y), 
                    bs_gpu, bs_t_gpu, K, N
                );
            }

            {
                const size_t global_tiles_x = (M + warp_size - 1) / warp_size;
                const size_t global_tiles_y = (N + warp_size - 1) / warp_size;

                const size_t global_work_size_x = global_tiles_x * work_group_side;
                const size_t global_work_size_y = global_tiles_y * work_group_side;

                matrix_multiplication_kernel.exec(
                    gpu::WorkSize(work_group_side, work_group_side, global_work_size_y, global_work_size_x), 
                    as_gpu, bs_t_gpu, cs_gpu, M, K, N
                );
            }

            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << gflops / t.lapAvg() << " GFlops" << std::endl;
    }

    cs_gpu.readN(cs.data(), M*N);

#ifdef WITH_CPU
    // Проверяем корректность результатов
    double diff_sum = 0;
    for (int i = 0; i < M * N; ++i) {
        double a = cs[i];
        double b = cs_cpu_reference[i];
        if (a != 0.0 || b != 0.0) {
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
#endif

    return 0;
}
