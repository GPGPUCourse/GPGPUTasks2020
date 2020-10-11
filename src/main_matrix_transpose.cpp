#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/device.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/opencl/device_info.h>

#include "cl/matrix_transpose_cl.h"

#include <cassert>
#include <vector>
#include <iostream>
#include <stdexcept>


int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    unsigned int M = 8192;
    unsigned int K = 8192;

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

    ocl::Kernel matrix_transpose_kernel(matrix_transpose, matrix_transpose_length, "matrix_transpose");
    matrix_transpose_kernel.compile();

    {
        ocl::DeviceInfo deviceInfo;
        deviceInfo.init(device.device_id_opencl);
        
        size_t work_group_side = 1;
        while ((work_group_side * work_group_side << 2) <= deviceInfo.max_workgroup_size) {
            work_group_side <<= 1;
        }

        const size_t warp_size = deviceInfo.warp_size != 0 ? deviceInfo.warp_size : deviceInfo.wavefront_width;
        assert(warp_size != 0 && "warp size is not zero");

        // we want tiles of WARP_SIZE x WARP_SIZE which will be filled line by line 
        // by whole warps and then subdivided into smaller WG_SIDE x WG_SIDE for transposition
        assert(work_group_side * work_group_side % warp_size == 0 && "wrong assumptions");
        assert(work_group_side < warp_size                        && "wrong assumptions");

        // adjust global work size accordingly
        const size_t global_tiles_x = (K + warp_size - 1) / warp_size;
        const size_t global_tiles_y = (M + warp_size - 1) / warp_size;

        const size_t global_work_size_x = global_tiles_x * work_group_side;
        const size_t global_work_size_y = global_tiles_y * work_group_side;

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            matrix_transpose_kernel.exec(
                gpu::WorkSize(work_group_side, work_group_side, global_work_size_x, global_work_size_y), 
                as_gpu, as_t_gpu, K, M
            );

            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << M*K/1000.0/1000.0 / t.lapAvg() << " millions/s" << std::endl;
    }

    as_t_gpu.readN(as_t.data(), M*K);

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
