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

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

int ceil_div(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
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

    int WORKGROUP_SIZE = 256;

    unsigned int n_ext = 0; // сколько нужно добавить внутренних узлов в префиксное дерево
    unsigned int ns[10]; // число узлов на всех уровнях
    unsigned int ns_size = 0; // высота всего дерева
    ns[0] = n;
    do {
        ns_size++;
        ns[ns_size] = ceil_div(ns[ns_size - 1], WORKGROUP_SIZE);
        n_ext += ns[ns_size];
    } while (ns[ns_size] > 1);
    ns_size++;
    unsigned int n_ext8 = n_ext * 8;

    gpu::gpu_mem_32u rs_gpu8, ns_gpu;
    gpu::gpu_mem_32u as_gpu[2];
    as_gpu[0].resizeN(n);
    as_gpu[1].resizeN(n);
    rs_gpu8.resizeN(n_ext8);
    ns_gpu.resizeN(ns_size);
    ns_gpu.writeN(ns, ns_size);
    std::vector<unsigned int> rs8(n_ext8, 0);

    {
        ocl::Kernel radix8(radix_kernel, radix_kernel_length, "radix8");
        ocl::Kernel partial_sum_main8(radix_kernel, radix_kernel_length, "partial_sum_main8");
        ocl::Kernel partial_sum8(radix_kernel, radix_kernel_length, "partial_sum8");
        ocl::Kernel partial_sum_gather8(radix_kernel, radix_kernel_length, "partial_sum_gather8");
        radix8.compile();
        partial_sum_main8.compile();
        partial_sum8.compile();
        partial_sum_gather8.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            int k = 0;
            as_gpu[k].writeN(as.data(), n);

            t.restart(); // Запускаем секундомер после прогрузки данных чтобы замерять время работы кернела, а не трансфер данных

            for (unsigned int shift = 0; shift < 32; shift += 3) {
                partial_sum_main8.exec(gpu::WorkSize(WORKGROUP_SIZE, n), as_gpu[k], rs_gpu8, n, n_ext, shift);

                for (unsigned int i = 1, offset = 0; i < ns_size - 1; offset += ns[i++]) {
                    partial_sum8.exec(gpu::WorkSize(WORKGROUP_SIZE, ns[i]), rs_gpu8, ns[i], offset, n_ext);
                }

                partial_sum_gather8.exec(gpu::WorkSize(WORKGROUP_SIZE, ns[1]), rs_gpu8, ns_gpu, ns_size, n_ext);

                radix8.exec(gpu::WorkSize(WORKGROUP_SIZE, n), as_gpu[k], as_gpu[(k + 1) % 2], rs_gpu8, n, n_ext, shift);
                k = (k + 1) % 2;
            }

            t.nextLap();
            as_gpu[k].readN(as.data(), n);
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}
