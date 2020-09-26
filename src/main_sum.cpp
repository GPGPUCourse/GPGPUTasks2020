#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/sum_cl.h"


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
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);
    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;

    unsigned int reference_sum = 0;
    unsigned int n = 100*1000*1000;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(42);
    for (int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(std::numeric_limits<unsigned int>::max() / n);
        reference_sum += as[i];
    }

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
        ocl::Kernel sum_simple(sum_kernel, sum_kernel_length, "sum_simple");
        sum_simple.compile(false);

        gpu::gpu_mem_32u as_gpu, out_gpu;
        as_gpu.resizeN(n);
        out_gpu.resizeN(1);
        as_gpu.writeN(as.data(), n);

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            // clear sum
            unsigned int tmp = 0;
            out_gpu.writeN(&tmp, 1);
            unsigned int workGroupSize = 256;
            sum_simple.exec(gpu::WorkSize(workGroupSize, n),
                            as_gpu, out_gpu, n);
            t.nextLap();
        }
        unsigned int sum = 0;
        out_gpu.readN(&sum, 1);
        EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
        std::cout << "GPU simple:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU simple:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        ocl::Kernel sum_simple(sum_kernel, sum_kernel_length, "sum_tree");
        sum_simple.compile(false);

        gpu::gpu_mem_32u as_gpu, out_gpu;
        as_gpu.resizeN(n);
        out_gpu.resizeN(1);
        as_gpu.writeN(as.data(), n);

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            // clear sum
            unsigned int tmp = 0;
            out_gpu.writeN(&tmp, 1);
            unsigned int workGroupSize = 256;
            sum_simple.exec(gpu::WorkSize(workGroupSize, n),
                        as_gpu, out_gpu, n);
            t.nextLap();
        }
        unsigned int sum = 0;
        out_gpu.readN(&sum, 1);
        EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
        std::cout << "GPU tree:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU tree:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        ocl::Kernel sum_simple(sum_kernel, sum_kernel_length, "sum_mult_vals");
        sum_simple.compile(false);

        gpu::gpu_mem_32u as_gpu, out_gpu;
        as_gpu.resizeN(n);
        out_gpu.resizeN(1);
        as_gpu.writeN(as.data(), n);

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            // clear sum
            unsigned int tmp = 0;
            out_gpu.writeN(&tmp, 1);
            unsigned int workGroupSize = 256;
            const unsigned int VALUES_PER_WORK_ITEM = 64;
            // Adjust globalWorkSize
            unsigned int globalWorkSize = (n + VALUES_PER_WORK_ITEM - 1) / VALUES_PER_WORK_ITEM;
            sum_simple.exec(gpu::WorkSize(workGroupSize, globalWorkSize),
                            as_gpu, out_gpu, n);
            t.nextLap();
        }
        unsigned int sum = 0;
        out_gpu.readN(&sum, 1);
        EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
        std::cout << "GPU mult_vals:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU mult_vals:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        ocl::Kernel sum_simple(sum_kernel, sum_kernel_length, "sum_tree_mult_vals");
        sum_simple.compile(false);

        gpu::gpu_mem_32u as_gpu, out_gpu;
        as_gpu.resizeN(n);
        out_gpu.resizeN(1);
        as_gpu.writeN(as.data(), n);

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            // clear sum
            unsigned int tmp = 0;
            out_gpu.writeN(&tmp, 1);
            unsigned int workGroupSize = 256;
            const unsigned int VALUES_PER_WORK_ITEM = 64;
            // Adjust globalWorkSize
            unsigned int globalWorkSize = (n + VALUES_PER_WORK_ITEM - 1) / VALUES_PER_WORK_ITEM;
            sum_simple.exec(gpu::WorkSize(workGroupSize, globalWorkSize),
                            as_gpu, out_gpu, n);
            t.nextLap();
        }
        unsigned int sum = 0;
        out_gpu.readN(&sum, 1);
        EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
        std::cout << "GPU tree mult vals:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU tree mult vals:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }
}