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
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

void merge_sort(gpu::gpu_mem_32f& gpu_data, ocl::Kernel& merge_kernel) {
    unsigned int workGroupSize = 256;
    unsigned int global_work_size = (gpu_data.number() + workGroupSize - 1) / workGroupSize * workGroupSize;

    gpu::gpu_mem_32f buffer_mem;
    buffer_mem.resizeN(gpu_data.number());
    int data_size = 2;
    merge_kernel.exec(gpu::WorkSize(workGroupSize, global_work_size),
                      gpu_data, buffer_mem, (unsigned int)gpu_data.number(), data_size);
    data_size = workGroupSize * 2;
    int count = 0;
    while (data_size <= gpu_data.number()) {
        ++count;
        if (count % 2 == 1) {
            merge_kernel.exec(gpu::WorkSize(workGroupSize, global_work_size),
                              buffer_mem, gpu_data, (unsigned int)gpu_data.number(), data_size);
        } else {
            merge_kernel.exec(gpu::WorkSize(workGroupSize, global_work_size),
                              gpu_data, buffer_mem, (unsigned int)gpu_data.number(), data_size);
        }
        data_size *= 2;
    }
    if (count % 2 != 1) {
        buffer_mem.copyToN(gpu_data, gpu_data.number());
    }
}


int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    unsigned int n = 32*1024*1024;
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

    gpu::gpu_mem_32f as_gpu;
    as_gpu.resizeN(n);
    {
        ocl::Kernel merge(merge_kernel, merge_kernel_length, "merge_sort");
        merge.compile();
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);
            t.restart(); // Запускаем секундомер после прогрузки данных чтобы замерять время работы кернела, а не трансфер данных
            unsigned int workGroupSize = 128;
            unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
            merge_sort(as_gpu, merge);
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;
        as_gpu.readN(as.data(), n);
    }
    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}
