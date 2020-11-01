#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/bitonic_cl.h"

#include <vector>
#include <iostream>
#include <stdexcept>
#include <climits>

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;

    unsigned int n = 32 * 1024 * 1024 + 10; //ok with non-Power of 2

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

    unsigned int maxPow = 2;

    while (maxPow < n) {
        maxPow *= 2;
    }

    as_gpu.resizeN(maxPow);
    std::vector<float> infinities(maxPow, MAXFLOAT);

    {
        ocl::Kernel bitonic(bitonic_kernel, bitonic_kernel_length, "bitonic");
        bitonic.compile();

        timer t;

        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(infinities.data(), maxPow);
            as_gpu.writeN(as.data(), n);

            unsigned int workGroupSize = 128;
            unsigned int global_work_size = (maxPow + workGroupSize - 1) / workGroupSize * workGroupSize;

            t.restart();

            unsigned int currSize = 2;

            while (currSize <= maxPow) {

                unsigned int sliceSize = currSize / 2;

                while (sliceSize > workGroupSize / 2) {

                    bitonic.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, sliceSize, currSize, maxPow,
                                 0);
                    sliceSize /= 2;
                }

                bitonic.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, sliceSize, currSize, maxPow, 1);
                currSize *= 2;
            }
            t.nextLap();
        }

        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;

        as_gpu.readN(as.data(), n);
    }

    for (int i = 0; i < n; ++i) {
        std::string answer(" -- GPU results should be equal to CPU results! different in pos " + std::to_string(i));
        EXPECT_THE_SAME(as[i], cpu_sorted[i], answer);
    }

    return 0;
}
