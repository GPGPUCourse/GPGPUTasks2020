#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/radix_cl.h"

#include <vector>
#include <iostream>
#include <stdexcept>


template<typename T>
void raiseFail(const T &a, const T &b, const std::string& message, const std::string& filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

void prefix(ocl::Kernel& prefix_kernel, ocl::Kernel& sum_kernel, gpu::gpu_mem_32i& array) {
    gpu::gpu_mem_32i next_gpu_array;

    unsigned int workGroupSize = 128;
    unsigned int global_work_size = (array.number() + workGroupSize - 1) / workGroupSize * workGroupSize;

    next_gpu_array.resizeN(global_work_size / workGroupSize);
    prefix_kernel.exec(gpu::WorkSize(workGroupSize, global_work_size), array, next_gpu_array,
                       (unsigned int)array.number());

    if (next_gpu_array.number() > 1) {
        prefix(prefix_kernel, sum_kernel, next_gpu_array);
    }
    sum_kernel.exec(gpu::WorkSize(workGroupSize, global_work_size), array, next_gpu_array,
                    (unsigned int)array.number());
}


int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    unsigned int n = 32 * 1024 * 1024;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<int>::max());
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::vector<unsigned int> cpu_sorted;
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

    gpu::gpu_mem_32u as_gpu;
    as_gpu.resizeN(n);

    {
        ocl::Kernel radix_set(radix_kernel, radix_kernel_length, "radix_set");
        ocl::Kernel radix_bits(radix_kernel, radix_kernel_length, "radix_bits");
        ocl::Kernel count_prefix(radix_kernel, radix_kernel_length, "count_prefix");
        ocl::Kernel count_sum(radix_kernel, radix_kernel_length, "count_sum");
        radix_set.compile();
        radix_bits.compile();
        count_prefix.compile();
        count_sum.compile();

        unsigned int workGroupSize = 256;
        unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        gpu::WorkSize wsize = gpu::WorkSize(workGroupSize, global_work_size);

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);
            t.stop();

            gpu::gpu_mem_32i zeroes;
            gpu::gpu_mem_32i ones;
            gpu::gpu_mem_32u another_as;
            zeroes.resizeN(n);
            ones.resizeN(n);
            another_as.resizeN(n);

            t.restart(); // Запускаем секундомер после прогрузки данных чтобы замерять время работы кернела, а не трансфер данных

            for (unsigned int i = 0; i < 32; ++i) {
                if (i % 2 == 0) {
                    radix_bits.exec(wsize, as_gpu, zeroes, ones, i, n);
                    prefix(count_prefix, count_sum, zeroes);
                    prefix(count_prefix, count_sum, ones);
                    radix_set.exec(wsize, as_gpu, another_as, zeroes, ones, i, n);
                } else {
                    radix_bits.exec(wsize, another_as, zeroes, ones, i, n);
                    prefix(count_prefix, count_sum, zeroes);
                    prefix(count_prefix, count_sum, ones);
                    radix_set.exec(wsize, another_as, as_gpu, zeroes, ones, i, n);
                }
            }
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
