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
void raiseFail(const T& a, const T& b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

int main(int argc, char** argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 100;
    std::cout << "We will run " << benchmarkingIters << " iterations in each configuration.\n";
    unsigned int n = 32 * 1024 * 1024; // Увы и ах. Я буду пользоваться тем, что тут степень двойки, потому что иначе будет очень неудобно.
                                       // Кроме того, n больше размера workGroup хотя бы в два раза. Что поделать!
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
        std::cout << "CPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;
    }
    gpu::gpu_mem_32f as_gpu;
    as_gpu.resizeN(n);
    for (unsigned work_group_size_log = 0; work_group_size_log <= 8; ++work_group_size_log)
    {
        std::cout << '\n';
        unsigned work_group_size = 1 << work_group_size_log;
        const std::string defines = "-D WORK_GROUP_SIZE=" + std::to_string(work_group_size) + " -D WORK_GROUP_SIZE_LOG=" + std::to_string(work_group_size_log);
        ocl::Kernel bitonic_left(bitonic_kernel, bitonic_kernel_length, "bitonic_left", defines);
        ocl::Kernel bitonic_middle(bitonic_kernel, bitonic_kernel_length, "bitonic_middle", defines);
        ocl::Kernel bitonic_right(bitonic_kernel, bitonic_kernel_length, "bitonic_right", defines);
        bitonic_left.compile();
        bitonic_middle.compile();
        bitonic_right.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);
            t.restart(); // Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных
            bitonic_left.exec(gpu::WorkSize(work_group_size, n >> 1), as_gpu);
            for (unsigned i = work_group_size_log + 1; (1 << i) < n; ++i)
            {
                for (unsigned j = i; j > work_group_size_log; --j)
                    bitonic_middle.exec(gpu::WorkSize(work_group_size, n >> 1), as_gpu, j, i);
                bitonic_right.exec(gpu::WorkSize(work_group_size, n >> 1), as_gpu, i);
            }

            t.nextLap();
        }
        std::cout << "GPU with workGroup = " << work_group_size << ": " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU with workGroup = " << work_group_size << ": " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;
        as_gpu.readN(as.data(), n);
    }
    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }
    system("pause");
    return 0;
}
