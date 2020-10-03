#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:22
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
    unsigned benchmarkingIters = 10;

    unsigned int reference_sum = 0;
    unsigned int n = 100*1000*1000;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(42);
    for (unsigned i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<unsigned int>::max() / n);
        reference_sum += as[i];
    }

    {
        timer t;
        for (unsigned iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            for (unsigned i = 0; i < n; ++i) {
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
        for (unsigned iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            #pragma omp parallel for reduction(+:sum)
            for (unsigned i = 0; i < n; ++i) {
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

        unsigned result = 0;
        auto result_gpu = gpu::gpu_mem_32u::createN(1);
        auto as_gpu = gpu::gpu_mem_32u::createN(n);
        as_gpu.writeN(as.data(), n);

        unsigned int workGroupSize = 128;
        unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        gpu::WorkSize work_size(workGroupSize, global_work_size);

        {
            std::string kernel_name = "sum1";
            ocl::Kernel sum(sum_kernel, sum_kernel_length, kernel_name);
            sum.compile();

            timer t;
            for (unsigned iter = 0; iter < benchmarkingIters; ++iter) {
                unsigned empty = 0;
                result_gpu.writeN(&empty, 1);
                sum.exec(work_size, result_gpu, as_gpu, n);

                result_gpu.readN(&result, 1);
                EXPECT_THE_SAME(reference_sum, result, "GPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU:     " << kernel_name << std::endl;
            std::cout << "GPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU:     " << n / 1e6 / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            std::string kernel_name = "sum2";
            ocl::Kernel sum(sum_kernel, sum_kernel_length, kernel_name);
            sum.compile();

            timer t;
            for (unsigned iter = 0; iter < benchmarkingIters; ++iter) {
                unsigned empty = 0;
                result_gpu.writeN(&empty, 1);
                sum.exec(work_size, result_gpu, as_gpu, n, 64);

                result_gpu.readN(&result, 1);
                EXPECT_THE_SAME(reference_sum, result, "GPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU:     " << kernel_name << std::endl;
            std::cout << "GPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU:     " << n / 1e6 / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            std::string kernel_name = "sum3";
            ocl::Kernel sum(sum_kernel, sum_kernel_length, kernel_name);
            sum.compile();

            timer t;
            for (unsigned iter = 0; iter < benchmarkingIters; ++iter) {
                unsigned empty = 0;
                result_gpu.writeN(&empty, 1);
                sum.exec(work_size, result_gpu, as_gpu, n, 64);

                result_gpu.readN(&result, 1);
                EXPECT_THE_SAME(reference_sum, result, "GPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU:     " << kernel_name << std::endl;
            std::cout << "GPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU:     " << n / 1e6 / t.lapAvg() << " millions/s" << std::endl;
        }

    }
}
