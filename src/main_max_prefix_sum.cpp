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
//        for (int i = 0; i < n; ++i) {
//            std::cout << as[i] << " ";
//            std::cout << std::endl;
//        }
        int reference_max_sum = -1;
        int reference_result = -1;
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

            std::string name = "getSums";
            ocl::Kernel kernel(max_prefix_sum_kernel, max_prefix_sum_kernel_length, name);
            kernel.compile(true);

            unsigned int workGroupSize = 128;


            gpu::WorkSize workSize = gpu::WorkSize(workGroupSize, 1);

            gpu::gpu_mem_32i as_gpu, sums_gpu;

            as_gpu.resizeN(n);
            as_gpu.writeN(as.data(), as.size());

            sums_gpu.resizeN(n);

            timer t;
            for (int iter = 0; iter < 1; ++iter) {

                std::vector<int> sums(n, 0);
                sums_gpu.writeN(sums.data(), n);

                kernel.exec(workSize, as_gpu, n, sums_gpu);

                sums_gpu.readN(sums.data(), n);

                int total_sum = - (1 << 29);
                int max_sum_index = -1;
                for (int i = 0; i < n; ++i) {
                    if (total_sum < sums[i]) {
                        total_sum = sums[i];
                        max_sum_index = i;
                    }
                }
                if (total_sum < 0) {
                    total_sum = 0;
                    max_sum_index = -1;
                }
                EXPECT_THE_SAME(reference_max_sum, total_sum, "GPU OpenCL sum result should be consistent!");
                EXPECT_THE_SAME(reference_result, max_sum_index + 1, "GPU OpenCL index result should be consistent!");
                t.nextLap();
            }
        }
        std::cout << "done" << std::endl;
    }
}
