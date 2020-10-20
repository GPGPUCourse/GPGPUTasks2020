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
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

void gpu_prefix(ocl::Kernel& prefix_kernel, ocl::Kernel& sum_kernel, gpu::gpu_mem_32i& array) {
    gpu::gpu_mem_32i next_gpu_array;

    unsigned int workGroupSize = 128;
    unsigned int global_work_size = (array.number() + workGroupSize - 1) / workGroupSize * workGroupSize;

    next_gpu_array.resizeN(global_work_size / workGroupSize);
    prefix_kernel.exec(gpu::WorkSize(workGroupSize, global_work_size), array, next_gpu_array, (unsigned int)array.number());

    if (next_gpu_array.number() > 1) {
        gpu_prefix(prefix_kernel, sum_kernel, next_gpu_array);
    }
    sum_kernel.exec(gpu::WorkSize(workGroupSize, global_work_size), array, next_gpu_array, (unsigned int)array.number());
}

void radix_sort(ocl::Kernel& radix_bits_kernel, ocl::Kernel& radix_set_kernel,
                ocl::Kernel& prefix_kernel, ocl::Kernel& sum_kernel, gpu::gpu_mem_32u& array) {
    unsigned int n = array.number();
    gpu::gpu_mem_32i zero_position;
    gpu::gpu_mem_32i one_position;
    gpu::gpu_mem_32u another_array;
    zero_position.resizeN(n);
    one_position.resizeN(n);
    another_array.resizeN(n);

    unsigned int workGroupSize = 256;
    unsigned int global_work_size = (array.number() + workGroupSize - 1) / workGroupSize * workGroupSize;
    for (unsigned int i = 0; i < 32; ++i) {
        if (i % 2 == 0) {
            radix_bits_kernel.exec(gpu::WorkSize(workGroupSize, global_work_size), array, zero_position, one_position, i, (unsigned int)array.number());
            gpu_prefix(prefix_kernel, sum_kernel, zero_position);
            gpu_prefix(prefix_kernel, sum_kernel, one_position);
            radix_set_kernel.exec(gpu::WorkSize(workGroupSize, global_work_size), array, another_array, zero_position, one_position, i, (unsigned int)array.number());
        } else {
            radix_bits_kernel.exec(gpu::WorkSize(workGroupSize, global_work_size), another_array, zero_position, one_position, i, (unsigned int)array.number());
            gpu_prefix(prefix_kernel, sum_kernel, zero_position);
            gpu_prefix(prefix_kernel, sum_kernel, one_position);
            radix_set_kernel.exec(gpu::WorkSize(workGroupSize, global_work_size), another_array, array, zero_position, one_position, i, (unsigned int)array.number());
        }
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
        ocl::Kernel prefix_kernel(radix_kernel, radix_kernel_length, "prefix_k");
        ocl::Kernel sum_kernel(radix_kernel, radix_kernel_length, "sum_k");
        radix_set.compile();
        radix_bits.compile();
        prefix_kernel.compile();
        sum_kernel.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart(); // Запускаем секундомер после прогрузки данных чтобы замерять время работы кернела, а не трансфер данных
            radix_sort(radix_bits, radix_set, prefix_kernel, sum_kernel, as_gpu);
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
