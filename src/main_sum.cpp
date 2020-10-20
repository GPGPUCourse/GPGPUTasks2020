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

#define VALS_IN_STEP 1024
#define WORKGROUP_SIZE 128

int main(int argc, char **argv)
{
    int benchmarkingIters = 1;

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

        std::string defines = "-D VALS_IN_STEP=" + std::to_string(VALS_IN_STEP)
                            + " -D WORKGROUP_SIZE=" + std::to_string(WORKGROUP_SIZE);
        ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum", defines);

        bool printLog = false;
        kernel.compile(printLog);

        gpu::gpu_mem_32u mem_gpu_const, mem_gpu0, mem_gpu1;
        mem_gpu_const.resizeN(n);
        mem_gpu0.resizeN(n);
        mem_gpu1.resizeN(n);
        mem_gpu_const.writeN(as.data(), n); // нужно чтобы не копировать массив между бенчмарками

        {
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                unsigned int current_n = n;
                bool from0to1 = false;
                bool firstCall = true;
                while (current_n > 1) {
                    const unsigned int next_n = gpu::divup(current_n, VALS_IN_STEP);
                    const unsigned int global_work_size = gpu::divup(next_n, WORKGROUP_SIZE) * WORKGROUP_SIZE;
                    from0to1 = !from0to1; // true в первой итерации
                    if (from0to1) {
                        if (firstCall) {
                            kernel.exec(gpu::WorkSize(WORKGROUP_SIZE, global_work_size), mem_gpu_const, mem_gpu1, current_n);
                            firstCall = false;
                        } else {
                            kernel.exec(gpu::WorkSize(WORKGROUP_SIZE, global_work_size), mem_gpu0, mem_gpu1, current_n);
                        }
                    } else {
                        kernel.exec(gpu::WorkSize(WORKGROUP_SIZE, global_work_size), mem_gpu1, mem_gpu0, current_n);
                    }
                    current_n = next_n;
                }

                unsigned int gpu_sum = 0; // ноль на случай n == 0
                if (from0to1) {
                    mem_gpu1.readN(&gpu_sum, 1);
                } else {
                    mem_gpu0.readN(&gpu_sum, 1);
                }
                EXPECT_THE_SAME(reference_sum, gpu_sum, "GPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
    }

    return 0;
}
