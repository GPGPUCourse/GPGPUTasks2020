#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/max_prefix_sum_cl.h"

template<typename T>
void raiseFail(const T &a, const T &b, const std::string& message, const std::string& filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

/*unsigned int getGlobalSize(unsigned int size, unsigned int workGroupSize) {
    return std::max(size, workGroupSize);
}*/

unsigned int getBufferSize(unsigned int size) {
    return std::max(size, 1u);
}


int main(int argc, char **argv)
{
    int benchmarkingIters = 100;
    int max_n = (1 << 24);

    gpu::Device device = gpu::chooseGPUDevice(argc, argv);
    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    ocl::Kernel gpu_prefix(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "prefix_sum");
    gpu_prefix.compile();

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
            /*timer t;
            unsigned int workGroupSize = 256;
            unsigned int values_per_item = 2;
            unsigned int global_work_size = (n + values_per_item - 1) / values_per_item;
            gpu::gpu_mem_32i as_gpu;
            as_gpu.resizeN(n);
            as_gpu.writeN(as.data(), n);

            unsigned int shrinker = workGroupSize * values_per_item;

            gpu::gpu_mem_32i bs_gpu;
            bs_gpu.resizeN(getBufferSize(n / shrinker));
            for (int i = 0; i < benchmarkingIters; ++i) {

                int count = 0;
                for (unsigned int gl_s = global_work_size; gl_s >= 1; gl_s /= shrinker) {
                    if (count % 2 == 0) {
                        gpu_prefix.exec(gpu::WorkSize(workGroupSize, gl_s),
                                        as_gpu, bs_gpu, as_gpu.size());
                        as_gpu.resizeN(getBufferSize(as_gpu.size() / shrinker / shrinker));
                    } else {
                        gpu_prefix.exec(gpu::WorkSize(workGroupSize, gl_s),
                                        bs_gpu, as_gpu, bs_gpu.size());
                        bs_gpu.resizeN(getBufferSize(bs_gpu.size() / shrinker / shrinker));
                    }
                    count++;
                }

                int prefix;
                if (count % 2 == 0) {
                    bs_gpu.readN(&prefix, 1);
                } else {
                    as_gpu.readN(&prefix, 1);
                }

                EXPECT_THE_SAME(reference_max_sum, prefix, "GPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;*/
        }
    }
}
