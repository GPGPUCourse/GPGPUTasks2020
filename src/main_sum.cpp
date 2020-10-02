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
        as[i] = (unsigned int) r.next(0, std::numeric_limits<unsigned int>::max() / n);
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
        unsigned int workGroupSize = 256;
        unsigned int sum_gpu = 0;
        unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;

        gpu::gpu_mem_32u as_gpu, cs_gpu;
        as_gpu.resizeN(n);
        cs_gpu.resizeN( 1);

        as_gpu.writeN(as.data(), n);

        ocl::Kernel sum_atomic(sum_kernel, sum_kernel_length, "sum_atomic");
        sum_atomic.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            sum_atomic.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, cs_gpu, n);
            cs_gpu.readN(&sum_gpu, 1);
            EXPECT_THE_SAME(reference_sum, sum_gpu, "CPU and atomic GPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "GPU_at:  " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU_at:  " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        unsigned int workGroupSize = 256;
        unsigned int sum_gpu = 0;
        unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;

        gpu::gpu_mem_32u as_gpu, cs_gpu;
        as_gpu.resizeN(n);
        cs_gpu.resizeN(1);

        as_gpu.writeN(as.data(), n);

        ocl::Kernel sum_tree(sum_kernel, sum_kernel_length, "sum_tree");
        sum_tree.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            sum_tree.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, cs_gpu, n);
            cs_gpu.readN(&sum_gpu, 1);
            EXPECT_THE_SAME(reference_sum, sum_gpu, "CPU and tree GPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "GPU_tr:  " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU_tr:  " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        unsigned int workGroupSize = 256;
        unsigned int sum_gpu = 0;
        unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;

        gpu::gpu_mem_32u as_gpu, bs_gpu, cs_gpu;
        as_gpu.resizeN(n);
        bs_gpu.resizeN(n);
        cs_gpu.resizeN((n - 1) / workGroupSize + 1);

        as_gpu.writeN(as.data(), n);

        ocl::Kernel sum(sum_kernel, sum_kernel_length, "sum");
        sum.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            gpu::gpu_mem_32u* from = &bs_gpu;
            gpu::gpu_mem_32u* to = &cs_gpu;
            as_gpu.copyToN(bs_gpu, n);
            for (unsigned int n_act = n; n_act > 1; n_act = (n_act - 1) / workGroupSize + 1) {
                sum.exec(gpu::WorkSize(workGroupSize, global_work_size), *from, *to, n_act);
                auto tmp = from;
                from = to;
                to = tmp;
            }
            (*from).readN(&sum_gpu, 1);
            EXPECT_THE_SAME(reference_sum, sum_gpu, "CPU and GPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "GPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }
    return 0;
}
