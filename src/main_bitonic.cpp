#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/bitonic_cl.h"

#include <vector>
#include <iostream>
#include <stdexcept>


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    unsigned int n = 32 * 1024 * 1024;
    std::vector<float> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.nextf();
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::vector<float> cpu_sorted;
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            cpu_sorted = as;
            std::sort(cpu_sorted.begin(), cpu_sorted.end());
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;
    }

    unsigned int pow2 = 0;
    while (1 << 1 + pow2++ <= n - 1);
    unsigned int n_pow2 = 1 << pow2;
    std::vector<float> as2(n_pow2, 0);
    for (int i = 0; i < n; i++) {
        as2[i] = as[i];
    }
    for (unsigned int i = n; i < n_pow2; i++) {
        as2[i] = CL_FLT_MAX;
    }

    gpu::gpu_mem_32f as2_gpu;
    as2_gpu.resizeN(n_pow2);

    {
        ocl::Kernel bitonic_local      (bitonic_kernel, bitonic_kernel_length, "bitonic_local");
        ocl::Kernel bitonic_global     (bitonic_kernel, bitonic_kernel_length, "bitonic_global");
        ocl::Kernel bitonic_global_tail(bitonic_kernel, bitonic_kernel_length, "bitonic_global_tail");
        bitonic_local.compile();

        int WORKGROUP_SIZE = 256;
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as2_gpu.writeN(as2.data(), n);

            t.restart(); // Запускаем секундомер после прогрузки данных чтобы замерять время работы кернела, а не трансфер данных

            gpu::WorkSize workSize(WORKGROUP_SIZE, n_pow2);

            bitonic_local.exec(workSize, as2_gpu);

            for (int block_size = WORKGROUP_SIZE * 2; block_size <= n_pow2; block_size *= 2) {
                for (int red_block_size = block_size; red_block_size >= WORKGROUP_SIZE * 2; red_block_size /= 2) {
                    bitonic_global.exec(workSize, as2_gpu, block_size, red_block_size);
                }
                bitonic_global_tail.exec(workSize, as2_gpu, block_size);
            }
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;

        as2_gpu.readN(as.data(), n);
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}
