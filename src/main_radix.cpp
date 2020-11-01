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
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)
#define SIZE_LENGTH 32

void prefixRec(ocl::Kernel &calculatePrefix, ocl::Kernel &addBuffer, gpu::gpu_mem_32u &tempArr, unsigned int size) {

    unsigned int workGroupSize = 128;
    unsigned int global_work_size = (size + workGroupSize - 1) / workGroupSize * workGroupSize;

    gpu::gpu_mem_32u temp;
    temp.resizeN(global_work_size / workGroupSize);

    calculatePrefix.exec(gpu::WorkSize(workGroupSize, global_work_size), tempArr, temp, size);

    if (global_work_size > workGroupSize) {
        prefixRec(calculatePrefix, addBuffer, temp, global_work_size / workGroupSize);
    }

    addBuffer.exec(gpu::WorkSize(workGroupSize, global_work_size), tempArr, temp, size);
}

int main(int argc, char **argv) {

    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;

    unsigned int n = 32 * 1024;

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
        std::cout << "CPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;
    }

    gpu::gpu_mem_32u as_gpu;
    as_gpu.resizeN(n);

    {
        ocl::Kernel radix(radix_kernel, radix_kernel_length, "radix");
        radix.compile();

        ocl::Kernel radixByBits(radix_kernel, radix_kernel_length, "radixByBits");
        radixByBits.compile();

        ocl::Kernel calculatePrefix(radix_kernel, radix_kernel_length, "calculatePrefix");
        calculatePrefix.compile();

        ocl::Kernel addBuffer(radix_kernel, radix_kernel_length, "addBuffer");
        addBuffer.compile();

        timer t;

        for (int iter = 0; iter < benchmarkingIters; ++iter) {

            as_gpu.writeN(as.data(), n);

            t.restart();

            unsigned int workGroupSize = 128;

            unsigned int size = n;
            unsigned int global_work_size = (size + workGroupSize - 1) / workGroupSize * workGroupSize;

            gpu::gpu_mem_32u pos0, pos1, buff;
            pos0.resizeN(size);
            pos1.resizeN(size);
            buff.resizeN(size);

            for (int i = 0; i < SIZE_LENGTH; ++i) {

                radixByBits.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, pos0, pos1, i, size);
                prefixRec(calculatePrefix, addBuffer, pos0, size);
                prefixRec(calculatePrefix, addBuffer, pos1, size);
                radix.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, buff, pos0, pos1, i, size);
                std::swap(as_gpu, buff);
            }
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;

        as_gpu.readN(as.data(), n);
    }

    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}
