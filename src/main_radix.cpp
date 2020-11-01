#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/radix_cl.h"

#include <vector>
#include <functional>
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

#define LOG_LEVEL 2

#if LOG_LEVEL >= 3
#   define PREVIEW 16
#else
#   define PREVIEW 0
#endif

template<typename T>
void preview(const std::vector<T> &a) {
    const size_t size = std::min(a.size(), (size_t) PREVIEW);
    for (size_t i = 0; i < size; ++i) {
        std::cout << a[i] << " ";
    }
    if (size > 0)
        std::cout << "\n";
}

template<typename T>
void preview(const gpu::shared_device_buffer_typed<T> &a) {
    const size_t size = std::min(a.number(), (size_t) PREVIEW);
    
    if (size > 0) {
        std::vector<T> tmp(size);
        a.readN(tmp.data(), size);

        preview(tmp);
    }
}

int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    unsigned int n = 1025; //32 * 1024 * 1024;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<int>::max());
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

    gpu::gpu_mem_32u a_cnts_gpu;
    a_cnts_gpu.resizeN(n + 1);

    gpu::gpu_mem_32u a_cnts_gpu_next;
    a_cnts_gpu_next.resizeN(n + 1);

    gpu::gpu_mem_32u as_gpu_next;
    as_gpu_next.resizeN(n);

    {
        unsigned int workGroupSize = 256;
        unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        
        std::string defines_string;
        for (const auto &define : std::initializer_list<std::string> {
            "LOCAL_SIZE=" + std::to_string(workGroupSize),
#ifdef LOG_LEVEL
            "LOG_LEVEL=" + std::to_string(LOG_LEVEL),
#endif
        }) {
            defines_string += " -D" + define;
        }

        ocl::Kernel radix_setup(radix_kernel, radix_kernel_length, "radix_setup", defines_string);
        radix_setup.compile();

        ocl::Kernel radix_gather(radix_kernel, radix_kernel_length, "radix_gather", defines_string);
        radix_gather.compile();

        ocl::Kernel radix_propagate(radix_kernel, radix_kernel_length, "radix_propagate", defines_string);
        radix_propagate.compile();

        ocl::Kernel radix_move(radix_kernel, radix_kernel_length, "radix_move", defines_string);
        radix_move.compile();

        const auto setup_buckets = [&](const unsigned int bit) {
#if LOG_LEVEL > 1
            std::cout << "start" << std::endl;
#endif
            radix_setup.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, n, a_cnts_gpu, bit);
        };

        const std::function<void(unsigned int, unsigned int)> prefix_sum = [&](const unsigned int work_size, const unsigned int step) {
#if LOG_LEVEL > 1
            std::cout << "\tstep " << step << std::endl;
#endif
            radix_gather.exec(gpu::WorkSize(workGroupSize, work_size), a_cnts_gpu, n, step);

            const auto next_step = step * workGroupSize;
            const auto next_work_size = (work_size + next_step - 1) / next_step;
            if (next_work_size > 1) {
                prefix_sum(next_work_size, next_step);
            }

            radix_propagate.exec(gpu::WorkSize(workGroupSize, global_work_size), a_cnts_gpu, n, next_step, a_cnts_gpu_next);
            a_cnts_gpu.swap(a_cnts_gpu_next);
#if LOG_LEVEL > 1
            preview(a_cnts_gpu);
            preview(a_cnts_gpu_next);
            std::cout << "\tstep " << step << " end" << std::endl;
#endif
        };

        const auto reorder = [&]() {
#if LOG_LEVEL > 1
            std::cout << "reorder" << std::endl;
#endif
            radix_move.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, n, a_cnts_gpu, as_gpu_next);
#if LOG_LEVEL > 1
            std::cout << "end" << std::endl;
#endif
        };

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart(); // Запускаем секундомер после прогрузки данных чтобы замерять время работы кернела, а не трансфер данных

            for (size_t bit = 0; bit < sizeof(unsigned int) * 8; ++bit) {
                setup_buckets(bit);
                prefix_sum(global_work_size, 1);
                reorder();

#if LOG_LEVEL > 0
                std::vector<unsigned int> as_current(n);
                as_gpu.readN(as_current.data(), n);
#if LOG_LEVEL > 2
                preview(as_current);
#endif

                std::vector<unsigned int> as_current_cnts(n + 1);
                a_cnts_gpu.readN(as_current_cnts.data(), n + 1);
#if LOG_LEVEL > 2
                preview(as_current_cnts);
#endif

                for (size_t i = 0; i < n; ++i) {
                    EXPECT_THE_SAME(i, as_current_cnts[i + 1], as_current_cnts[i] + ((as_current[i] >> bit) & 1), "partial sums should be correct");
                }
#endif

                as_gpu.swap(as_gpu_next);

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
