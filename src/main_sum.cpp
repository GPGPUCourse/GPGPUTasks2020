#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/sum_cl.h"

#ifdef __CLION_IDE__

#include "cl/sum.cl"

#endif


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
    int benchmarkingIters = 10;

    unsigned int referenceSum = 0;
    unsigned int n = 100 * 1000 * 1000;
    unsigned int workGroupSize = 128;
    unsigned int globalWorkSize = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
    std::vector<unsigned int> as(globalWorkSize, 0);
    FastRandom r(42);
    for (int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<unsigned int>::max() / n);
        referenceSum += as[i];
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(referenceSum, sum, "CPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU:     " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
#pragma omp parallel for reduction(+:sum)
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(referenceSum, sum, "CPU OpenMP result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU OMP: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU OMP: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();
        ocl::Kernel sum(sum_kernel, sum_kernel_length, "sum");
        sum.compile();
        auto sumGpu = gpu::gpu_mem_32u::createN(1);
        auto asGpu = gpu::gpu_mem_32u::createN(globalWorkSize);
        asGpu.writeN(as.data(), globalWorkSize);
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sumResult = 0;
            sumGpu.writeN(&sumResult, 1);
            sum.exec(gpu::WorkSize(workGroupSize, globalWorkSize), asGpu, sumGpu);
            sumGpu.readN(&sumResult, 1);
            EXPECT_THE_SAME(referenceSum, sumResult, "CPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "GPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU:     " << (globalWorkSize / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }
}
