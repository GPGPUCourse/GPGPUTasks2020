#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

// Different summation kernels
#include "cl/sum_cl.h"
#include "cl/sum_less_atomic_cl.h"
#include "cl/sum_recursive_cl.h"

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

void run_kernel(int argc, char **argv, unsigned int n,
                std::vector<unsigned int>& as,
                unsigned int reference_sum,
                unsigned int benchmarkingIters,
                const std::string kernel_type,
                const char* kernel_source,
                size_t kernel_length) {

    {
        std::cout << "Running kernel: " << kernel_type << std::endl;

        gpu::Device device = gpu::chooseGPUDevice(argc, argv);

        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();
        {
            ocl::Kernel sum(kernel_source, kernel_length, kernel_type);
            sum.compile();

            // Set-up GPU memory buffer
            gpu::gpu_mem_32u gpu_sum_buffer;
            gpu_sum_buffer.resizeN(n);
            gpu_sum_buffer.writeN(as.data(), n);

            gpu::gpu_mem_32u gpu_sum_result;
            gpu_sum_result.resizeN(1);

            // Set-up NDRange
            const unsigned int workGroupSize = 128;
            unsigned int global_work_size;

            timer t;
            if(kernel_type == "sum_default" || kernel_type == "sum_less_atomic"){
                const unsigned int data_per_workitem = 8;

                if (kernel_type == "sum_default")
                    global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
                else
                    global_work_size = (n + data_per_workitem - 1) / data_per_workitem;

                // Kernel runs
                for (int i = 0; i < benchmarkingIters; ++i) {
                    unsigned int sum_value = 0;
                    gpu_sum_result.writeN(&sum_value, 1);
                    sum.exec(gpu::WorkSize(workGroupSize, global_work_size),
                             gpu_sum_buffer, n, gpu_sum_result);

                    gpu_sum_result.readN(&sum_value, 1);
                    EXPECT_THE_SAME(reference_sum, sum_value, "GPU result should be consistent!");
                    t.nextLap();
                }
            }
            else if(kernel_type == "sum_recursive"){
                std::cout << "Recursive call" << "\n";

                // Reset buffer
                gpu::gpu_mem_32u gpu_reset_sum_buffer;
                gpu_reset_sum_buffer.resizeN(n);
                gpu_sum_buffer.copyToN(gpu_reset_sum_buffer, n);

                for (int i = 0; i < benchmarkingIters; ++i) {
                    global_work_size = ((n + workGroupSize - 1) / workGroupSize) * workGroupSize;

                    t.stop(); // Each benchmark requires initial buffer reallocation
                    gpu_reset_sum_buffer.copyToN(gpu_sum_buffer, n);
                    t.start();

                    gpu::gpu_mem_32u swap_sum_buffer; // result buffer for swapping
                    swap_sum_buffer.resizeN(global_work_size / workGroupSize);

                    unsigned int remain_size = global_work_size;
                    unsigned int prev_remain_size;
                    while (true){

                        prev_remain_size = remain_size; // Save real remaining size
                        remain_size = ((remain_size + workGroupSize - 1) / workGroupSize) * workGroupSize; // Pad to match workGroupSize

                        sum.exec(gpu::WorkSize(workGroupSize, remain_size),
                                 gpu_sum_buffer, swap_sum_buffer, prev_remain_size);

                        remain_size /= workGroupSize;

                        gpu_sum_buffer.swap(swap_sum_buffer);
                        if(remain_size == 1){
                            break;
                        }
                    }

                    unsigned int sum_value = 0;
                    gpu_sum_buffer.readN(&sum_value, 1);

                    unsigned int sum_value_2 = 0;
                    swap_sum_buffer.readN(&sum_value_2, 1);

                    EXPECT_THE_SAME(reference_sum, sum_value, "GPU result should be consistent!");
                    t.nextLap();
                }
            }
            else{
                throw std::runtime_error("INVALID KERNEL OPTION");
            }

            // Compute metrics
            std::cout << kernel_type << " GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << kernel_type << " GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
    }

}


int main(int argc, char **argv)
{
    int benchmarkingIters = 10;

    unsigned int reference_sum = 0;
    unsigned int n = 100*1000*1000;
    // Another tests
    // unsigned int n = 4096;
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
        run_kernel(argc, argv, n, as, reference_sum,
                   benchmarkingIters, "sum_default", sum_kernel, sum_kernel_length);

        run_kernel(argc, argv, n, as, reference_sum,
                   benchmarkingIters, "sum_less_atomic",
                   sum_less_atomic_kernel, sum_less_atomic_kernel_length);

        run_kernel(argc, argv, n, as, reference_sum,
                   benchmarkingIters, "sum_recursive",
                   sum_recursive_kernel, sum_recursive_kernel_length);
    }
}
