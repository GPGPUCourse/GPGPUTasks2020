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
#include <climits>


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

void print(std::vector<unsigned int> ss) {
    for (auto s : ss) {
        std::cout << s << "; ";
    }
    std::cout << std::endl;
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

    gpu::gpu_mem_32u as_gpu, bs_gpu;
    as_gpu.resizeN(n);
    bs_gpu.resizeN(n);

    {
        unsigned int WGS = 256;

        gpu::gpu_mem_32u* from = &as_gpu;
        gpu::gpu_mem_32u* to = &bs_gpu;

        gpu::gpu_mem_32u prefix_sums_gpu;
        prefix_sums_gpu.resizeN((n + WGS - 1) / WGS);

        ocl::Kernel radix(radix_kernel, radix_kernel_length, "radix");
        radix.compile();

        ocl::Kernel totalFalses(radix_kernel, radix_kernel_length, "totalFalses");
        totalFalses.compile();

        ocl::Kernel prefixSum(radix_kernel, radix_kernel_length, "prefixSum");
        totalFalses.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart(); // Запускаем секундомер после прогрузки данных чтобы замерять время работы кернела, а не трансфер данных
            for (unsigned int bit_number = 0; bit_number < sizeof(unsigned int) * CHAR_BIT; ++bit_number) {
                unsigned int global_work_size = (n + WGS - 1) / WGS * WGS;
                totalFalses.exec(gpu::WorkSize(WGS, global_work_size),
                                 *from, prefix_sums_gpu, n, bit_number);

                unsigned int small_global_work_size = ((n + WGS - 1) / WGS + WGS - 1) / WGS * WGS;

                // сперва выполняем суммирование для размера, соответствующего количеству потоков в WGS
                prefixSum.exec(gpu::WorkSize(WGS, small_global_work_size),
                               prefix_sums_gpu, (n + WGS - 1) / WGS, WGS / 2);
                // затем выполняем остальное суммирование, если необходимо
                for (unsigned int j = WGS; j < (n + WGS - 1) / WGS; j *= 2) {
                    prefixSum.exec(gpu::WorkSize(WGS, small_global_work_size),
                                   prefix_sums_gpu, (n + WGS - 1) / WGS, j);
                }

                // сама сортировка
                radix.exec(gpu::WorkSize(WGS, global_work_size),
                           *from, *to, prefix_sums_gpu, n, bit_number);

                auto tmp = from;
                from = to;
                to = tmp;
            }
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;

        (*from).readN(as.data(), n);
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}
