#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/max_prefix_sum_cl.h"


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


void gpu_prefix(ocl::Kernel& prefix_kernel, ocl::Kernel& sum_kernel, gpu::gpu_mem_32i& array) {
    gpu::gpu_mem_32i next_gpu_array;

    unsigned int workGroupSize = 128;
    unsigned int global_work_size = (array.number() + workGroupSize - 1) / workGroupSize * workGroupSize;

    next_gpu_array.resizeN(global_work_size / workGroupSize);
    prefix_kernel.exec(gpu::WorkSize(workGroupSize, global_work_size), array, next_gpu_array, (unsigned int)array.number());

    if (next_gpu_array.number() > 1) {
        gpu_prefix(prefix_kernel, sum_kernel, next_gpu_array);
    }
    sum_kernel.exec(gpu::WorkSize(workGroupSize, global_work_size), array, next_gpu_array, (unsigned int)array.number());
}


int main(int argc, char **argv)
{
    int benchmarkingIters = 10;
    int max_n = (1 << 24);

    for (int n = 2; n <= max_n; n *= 2) {
        std::cout << "______________________________________________" << std::endl;
        int values_range = std::min(1023, std::numeric_limits<int>::max() / n);
        std::cout << "n=" << n << " values in range: [" << (-values_range) << "; " << values_range << "]" << std::endl;

        std::vector<int> as(n, 0);
        FastRandom r(n);
        for (int i = 0; i < n; ++i) {
            as[i] = (unsigned int) r.next(-values_range, values_range);
        }
        std::cout << std::endl;

        int reference_max_sum;
        int reference_result;
        {
            int max_sum = 0;
            int sum = 0;
            int result = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
                if (sum > max_sum) {
                    max_sum = sum;
                    result = i + 1;
                }
            }
            reference_max_sum = max_sum;
            reference_result = result;
        }
        std::cout << "Max prefix sum: " << reference_max_sum << " on prefix [0; " << reference_result << ")" << std::endl;

        {
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                int max_sum = 0;
                int sum = 0;
                int result = 0;
                for (int i = 0; i < n; ++i) {
                    sum += as[i];
                    if (sum > max_sum) {
                        max_sum = sum;
                        result = i + 1;
                    }
                }
                EXPECT_THE_SAME(reference_max_sum, max_sum, "CPU result should be consistent!");
                EXPECT_THE_SAME(reference_result, result, "CPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "CPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            // TODO: implement on OpenCL
            gpu::Device device = gpu::chooseGPUDevice(argc, argv);
            gpu::Context context;
            context.init(device.device_id_opencl);
            context.activate();

            gpu::gpu_mem_32i arr_mem;
            arr_mem.resizeN(n);
            arr_mem.writeN(as.data(), n);

            ocl::Kernel prefix_k(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "prefix_k");
            ocl::Kernel sum_k(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "sum_k");
            prefix_k.compile();
            sum_k.compile();

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                arr_mem.writeN(as.data(), n);
                gpu_prefix(prefix_k, sum_k, arr_mem);
                t.nextLap();
            }

            int prefix_max_sum = 0;
            int result_max = 0;

            std::vector<int> buffer_cpu(n);
            arr_mem.readN(buffer_cpu.data(), buffer_cpu.size());
            for (int i = 0; i < buffer_cpu.size(); ++i) {
                if (buffer_cpu[i] > prefix_max_sum) {
                    prefix_max_sum = buffer_cpu[i];
                    result_max = i + 1;
                }
            }

            EXPECT_THE_SAME(reference_max_sum, prefix_max_sum, "GPU result should be consistent!");
            EXPECT_THE_SAME(reference_result, result_max, "GPU result should be consistent!");

            std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
    }
}
