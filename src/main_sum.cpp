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
        // TODO: implement on OpenCL
         gpu::Device device = gpu::chooseGPUDevice(argc, argv);
         gpu::Context context;
         context.init(device.device_id_opencl);
         context.activate();

         unsigned int wg_size = 32;

         std::string defines = "-D WORK_GROUP_SIZE=" + std::to_string(wg_size);
         ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum_tree", defines);

         bool print_log{false};
         kernel.compile(print_log);

         unsigned int global_work_size = (n + wg_size - 1) / wg_size * wg_size;
         timer t;
         gpu::gpu_mem_32u as_gpu;
         as_gpu.resizeN(global_work_size);
         as_gpu.writeN(as.data(), n);

        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            gpu::gpu_mem_32u bs_gpu;
            bs_gpu.resizeN(n);

            t.stop(); // there is really no need to take this time into account
            as_gpu.copyToN(bs_gpu, n);
            t.start();

            unsigned int sum = 0;
            gpu::gpu_mem_32u sum_gpu;
            for (unsigned int size = global_work_size; size > 1; size = (size + wg_size - 1) / wg_size * wg_size / wg_size) {
                const auto sum_size = size / wg_size;
                sum_gpu.resizeN(0 < sum_size ? sum_size : 1);
                kernel.exec(gpu::WorkSize(wg_size, size), bs_gpu, sum_gpu, size);

                bs_gpu.swap(sum_gpu);
            }

            bs_gpu.readN(&sum, 1);
            EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "GPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }
}
