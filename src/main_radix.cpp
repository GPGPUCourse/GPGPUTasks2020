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
void raiseFail(size_t i, const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << " at " << i << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(i, a, b, message) raiseFail(i, a, b, message, __FILE__, __LINE__)

#define PREVIEW 16

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

    int benchmarkingIters = 1;//10;
    unsigned int n = 16;// 32 * 1024 * 1024;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, 100);// std::numeric_limits<int>::max());
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;
    preview(as);

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
    preview(cpu_sorted);

    gpu::gpu_mem_32u as_gpu;
    as_gpu.resizeN(n);

    gpu::gpu_mem_32u counts_gpu;
    counts_gpu.resizeN(n + 1);

    gpu::gpu_mem_32u bs_gpu;
    bs_gpu.resizeN(n);

    {
        unsigned int workGroupSize = 256;
        unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        
        const auto local_size_def = "-DLOCAL_SIZE=" + std::to_string(workGroupSize);

        ocl::Kernel radix_setup(radix_kernel, radix_kernel_length, "radix_setup", local_size_def);
        radix_setup.compile();

        ocl::Kernel radix_gather(radix_kernel, radix_kernel_length, "radix_gather", local_size_def);
        radix_gather.compile();

        ocl::Kernel radix_propagate(radix_kernel, radix_kernel_length, "radix_propagate", local_size_def);
        radix_propagate.compile();

        ocl::Kernel radix_move(radix_kernel, radix_kernel_length, "radix_move", local_size_def);
        radix_move.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart(); // Запускаем секундомер после прогрузки данных чтобы замерять время работы кернела, а не трансфер данных

            for (size_t bit = 0; bit < sizeof(unsigned int) * 8; ++bit) {
                radix_setup.exec(
                    gpu::WorkSize(workGroupSize, global_work_size),
                    bit % 2 ? bs_gpu : as_gpu, n, counts_gpu, (unsigned int) bit
                );

                for (size_t step = 1; step <= n + 1; step *= workGroupSize) {
                    radix_gather.exec(
                        gpu::WorkSize(workGroupSize, global_work_size),
                        counts_gpu, n, (unsigned int) step
                    );
                }

                radix_propagate.exec(
                    gpu::WorkSize(workGroupSize, global_work_size),
                    counts_gpu, n
                );
                
                counts_gpu.readN(as.data(), n);
                preview(as);

                radix_move.exec(
                    gpu::WorkSize(workGroupSize, global_work_size),
                    bit % 2 ? bs_gpu : as_gpu, n, counts_gpu, bit % 2 ? as_gpu : bs_gpu
                );

                (bit % 2 ? as_gpu : bs_gpu).readN(as.data(), n);
                preview(as);
            }
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;

        as_gpu.readN(as.data(), n);
    }
    preview(as);

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(i, as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }
    return 0;
}
