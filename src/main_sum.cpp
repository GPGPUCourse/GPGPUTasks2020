#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>

#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/sum_cl.h"
// #include <assert>

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

int main(int argc, char **argv)
{
    int benchmarkingIters = 10;

    unsigned int reference_sum = 0;
    // unsigned int n = 1*1000*1000;
    unsigned int n = 128 * 64 * 32;
    std::vector<unsigned int> as(n);
    long long res = 0;
    FastRandom r(42);
    for (int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<unsigned int>::max() / n);
        res += as[i];
        reference_sum += as[i];
    }
    assert(res == reference_sum);

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            #pragma omp parallel for reduction(+:sum)
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU OpenMP result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU OMP: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU OMP: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        gpu::Device device = gpu::chooseGPUDevice(argc, argv);
        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();
     
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            gpu::gpu_mem_32u gpuA;
            gpuA.resizeN(n);
            gpuA.writeN(as.data(), n);

            ocl::Kernel summa(sum_kernel, sum_kernel_length, "summa");
            summa.compile();

            unsigned int res = 0;
            gpu::gpu_mem_32u gpuRes;
            gpuRes.resizeN(1);
            gpuRes.writeN(&res, 1);
            unsigned int workGroupSize = 128;
            unsigned int rangePerWorkItem = 64;
            unsigned int globalWorkSize = (n + workGroupSize - 1) / workGroupSize * workGroupSize / rangePerWorkItem;

            summa.exec(gpu::WorkSize(workGroupSize, globalWorkSize), gpuA, gpuRes, n);

            gpuRes.readN(&res, 1);
            EXPECT_THE_SAME(reference_sum, res, "OpenCL result should be consistent!");
            break;
            t.nextLap();
        }
        std::cout << "OpenCL: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "OpenCL: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }
}
