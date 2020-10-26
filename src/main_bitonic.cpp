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
void raiseFail(size_t i, const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << " at " << i << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(i, a, b, message) raiseFail(i, a, b, message, __FILE__, __LINE__)

// #define PREVIEW 16
#define MAX 179179

template<typename T>
void preview(const std::vector<T> &a) {
#ifdef PREVIEW
    for (size_t i = 0; i < std::min(a.size(), (size_t) PREVIEW); ++i) {
        std::cout << a[i] << " ";
    }
    std::cout << "\n";
#endif
}

int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    size_t n = 32 * 1024 * 1024;
    std::vector<float> as(n, 0);
    FastRandom r(n);
    for (size_t i = 0; i < n; ++i) {
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
    preview(cpu_sorted);

    const size_t workGroupSize = 256;
    const size_t initialStep = 8;

    const size_t global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
            
    gpu::gpu_mem_32f as_gpu;
    as.resize(global_work_size, MAX); // some temporary values for sorting
    as_gpu.resizeN(global_work_size);

    {
        ocl::Kernel bitonic_local(
            bitonic_kernel, bitonic_kernel_length, "bitonic", 
            "-DLOCAL_SIZE=" + std::to_string(workGroupSize)
        );
        bitonic_local.compile();

        ocl::Kernel bitonic_global(bitonic_kernel, bitonic_kernel_length, "bitonic");
        bitonic_global.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), global_work_size);

            t.restart(); // Запускаем секундомер после прогрузки данных чтобы замерять время работы кернела, а не трансфер данных

            bitonic_local.exec(
                gpu::WorkSize(workGroupSize, global_work_size), 
                as_gpu, (unsigned int) global_work_size, (unsigned int) initialStep, 0
            );
            //                       V -- boxes up to workGroupSize were already sorted with local memory
            for (size_t globalStep = initialStep; (1 << globalStep) <= 2 * global_work_size; ++globalStep) {
                for (size_t subStep = globalStep; subStep > 0; --subStep) {
                    bitonic_global.exec(
                        gpu::WorkSize(workGroupSize, global_work_size / 2), 
                        as_gpu, (unsigned int) global_work_size, (unsigned int) globalStep, (unsigned int) subStep
                    );
                }
            }
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;

        as_gpu.readN(as.data(), n);
        preview(as);
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(i, as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }
    return 0;
}
