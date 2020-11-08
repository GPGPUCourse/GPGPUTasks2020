#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/merge_cl.h"

#include <vector>
#include <iostream>
#include <stdexcept>


template<typename T>
void raiseFail(const size_t i, const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << " at " << i << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(i, a, b, message) raiseFail(i, a, b, message, __FILE__, __LINE__)

// #define DEBUG 8

int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

#ifdef DEBUG
    int benchmarkingIters = 1;
    unsigned int n = DEBUG;
#else
    int benchmarkingIters = 10;
    unsigned int n = 32*1024*1024;
#endif

    std::vector<float> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {

#ifdef DEBUG
        as[i] = r.next(0, 16);
#else
        as[i] = r.nextf();
#endif
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;
#ifdef DEBUG
    for (const auto &x: as) {
        std::cout << x << " ";
    }
    std::cout << std::endl;
#endif

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
    
    {
        const unsigned int workGroupSize = 256;
        const unsigned int global_work_size = [n, workGroupSize]() { // I'm tired of checking bounds all the time, 
            unsigned int power = 1;                                  // so finding first power of 2 not smaller than needed
            while (power < std::max(n, workGroupSize)) {             // worst case: n is 2^k - 1 for some k then the result would be 2^(k + 1)
                power <<= 1;                                         //             then at most ~2n => we would experience ~2x slowdown  
            }                                                        //             as we have n log^2(n) complexity

            return power;
        }();

        { // padding
            const auto maximum = *std::max_element(as.begin(), as.end());
            std::cout << "padding the array with " << maximum << std::endl;
            as.resize(global_work_size, maximum);
        }

        gpu::gpu_mem_32f as_gpu;
        as_gpu.resizeN(global_work_size);
        
        gpu::gpu_mem_32f bs_gpu;
        bs_gpu.resizeN(global_work_size);

        const std::string defines = "-DLOCAL_SIZE=" + std::to_string(workGroupSize);

        ocl::Kernel merge_local(merge_kernel, merge_kernel_length, "merge", defines + " -DUSE_LOCAL");
        merge_local.compile();

        ocl::Kernel merge_global(merge_kernel, merge_kernel_length, "merge", defines);
        merge_global.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), global_work_size);
            t.restart(); // Запускаем секундомер после прогрузки данных чтобы замерять время работы кернела, а не трансфер данных

            merge_local.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, as_gpu, n, 0);
            
            for (unsigned int step = workGroupSize; step < n; step <<= 1) {
                merge_global.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, bs_gpu, n, step);
                as_gpu.swap(bs_gpu);
            }

            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (global_work_size/1000/1000) / t.lapAvg() << " millions/s" << std::endl;
        as_gpu.readN(as.data(), n);
#ifdef DEBUG
        as.resize(n);
        for (const auto &x: as) {
            std::cout << x << " ";
        }
        std::cout << std::endl;
#endif
    }
    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(i, as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}
