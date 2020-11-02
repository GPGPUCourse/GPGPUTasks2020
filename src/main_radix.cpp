#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/radix_cl.h"
#include "cl/prefix_cl.h"

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

void print(const gpu::gpu_mem_32u& s, int n, int level = -1){
    std::vector<unsigned int> v(n);
    s.readN(v.data(), n);
    for (int x : v)
        std::cout << (level == -1 ? x : ((x >> level) & 1)) << ' ';
    std::cout << std::endl;
}

unsigned int runRecursion(ocl::Kernel& prefix, unsigned int workGroupSize,
                        gpu::gpu_mem_32u& prefixSums,
                        unsigned int shift, unsigned int n)
{
    if (n == 0)
        return 0;
    if (n == 1){
        unsigned int res;
        prefixSums.readN(&res, 1);
        // std::cout << res << std::endl;
        return res;
    }
    gpu::gpu_mem_32u prefixSumsBlocks;
    unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
    unsigned int newN = (n - 1) / workGroupSize + 1;
    prefixSumsBlocks.resizeN(newN);

    prefix.exec(gpu::WorkSize(workGroupSize, global_work_size),
        prefixSums, prefixSumsBlocks, 1, shift, n);

    auto res = runRecursion(prefix, workGroupSize, prefixSumsBlocks, shift, newN);

    // std::cout << "here " << newN << std::endl;
    // print(prefixSums, n);
    // print(prefixSumsBlocks, newN);

    prefix.exec(gpu::WorkSize(workGroupSize, global_work_size),
        prefixSums, prefixSumsBlocks, 2, shift, n);    

    // print(prefixSums, n);
    return res;
}

unsigned int calcPrefixSums(ocl::Kernel& prefix, unsigned int workGroupSize,
                        const gpu::gpu_mem_32u& s, gpu::gpu_mem_32u& prefixSums,
                        unsigned int shift, unsigned int n)
{
    unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
    prefix.exec(gpu::WorkSize(workGroupSize, global_work_size),
        s, prefixSums, 0, shift, n);
    return runRecursion(prefix, workGroupSize, prefixSums, shift, n);
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

    gpu::gpu_mem_32u s[2], prefixSums;
    s[0].resizeN(n);
    s[1].resizeN(n);

    {
        ocl::Kernel radix(radix_kernel, radix_kernel_length, "radix");
        radix.compile();

        ocl::Kernel prefix(prefix_kernel, prefix_kernel_length, "prefix");
        prefix.compile();

        prefixSums.resizeN(n);
        prefixSums.writeN(std::vector<unsigned int>(n).data(), n);

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            s[0].writeN(as.data(), n);

            t.restart(); // Запускаем секундомер после прогрузки данных чтобы замерять время работы кернела, а не трансфер данных

            unsigned int workGroupSize = 128;
            unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;

            for (int shift = 0; shift < 32; shift++) {

                auto zeroCount = calcPrefixSums(prefix, workGroupSize, s[shift & 1], prefixSums, shift, n);
                radix.exec(gpu::WorkSize(workGroupSize, global_work_size),
                           s[shift & 1], s[(shift & 1) ^ 1], prefixSums,
                           zeroCount, shift, n);
            }
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;

        s[0].readN(as.data(), n);
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}
