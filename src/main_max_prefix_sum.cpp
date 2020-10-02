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


int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);
    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

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
            unsigned int workGroupSize = 256;
            int max_prefix_sum_gpu = 0;
            unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;

            gpu::gpu_mem_32i initial_gpu, sum_a_gpu, max_a_gpu, sum_b_gpu, max_b_gpu;
            initial_gpu.resizeN(n);
            sum_a_gpu.resizeN(n);
            max_a_gpu.resizeN(n);
            sum_b_gpu.resizeN((n - 1) / workGroupSize + 1);
            max_b_gpu.resizeN((n - 1) / workGroupSize + 1);

            initial_gpu.writeN(as.data(), n);

            ocl::Kernel max_prefix_sum(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "max_prefix_sum");
            max_prefix_sum.compile();

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                gpu::gpu_mem_32i* from_max = &max_a_gpu;
                gpu::gpu_mem_32i* from_sum = &sum_a_gpu;
                gpu::gpu_mem_32i* to_max = &max_b_gpu;
                gpu::gpu_mem_32i* to_sum = &sum_b_gpu;
                initial_gpu.copyToN(max_a_gpu, n);
                initial_gpu.copyToN(sum_a_gpu, n);
                for (unsigned int n_act = n; n_act > 1; n_act = (n_act - 1) / workGroupSize + 1) {
                    max_prefix_sum.exec(gpu::WorkSize(workGroupSize, global_work_size), *from_sum, *from_max, *to_sum, *to_max, n_act);
                    auto tmp = from_max;
                    from_max = to_max;
                    to_max = tmp;
                    tmp = from_sum;
                    from_sum = to_sum;
                    to_sum = tmp;
                }
                (*from_max).readN(&max_prefix_sum_gpu, 1);
                EXPECT_THE_SAME(reference_max_sum, max_prefix_sum_gpu, "CPU and GPU result should be consistent!");
                t.nextLap();
            }
        }
    }
}
