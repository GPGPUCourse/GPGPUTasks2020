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

#define WORK_GROUP_SIZE 256

unsigned int ceil_div(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

int min(int a, int b) {
    return (a < b) ? a : b;
}

int main(int argc, char **argv)
{
    int benchmarkingIters = 10;
    int max_n = (1 << 24);

    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

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
            std::cout << "CPU:       " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "CPU:       " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                gpu::gpu_mem_32i gs_gpu;
                gpu::gpu_mem_32i ps_gpu;
                gpu::gpu_mem_32u pi_gpu;
                gs_gpu.resizeN(n);
                ps_gpu.resizeN(n);
                pi_gpu.resizeN(n);
                gs_gpu.writeN(as.data(), n);

                ocl::Kernel kernel_pref(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "max_prefix_sum_inplace");
                kernel_pref.compile();

                unsigned int density = 1;
                for (unsigned int len = n; len > 1; len = ceil_div(len, WORK_GROUP_SIZE)) {
                    gpu::WorkSize workSize(WORK_GROUP_SIZE, len);
                    kernel_pref.exec(workSize, gs_gpu, ps_gpu, pi_gpu, density, len);
                    density *= WORK_GROUP_SIZE;
                }
                int max_sum = 0;
                unsigned int result = 0;
                ps_gpu.readN(&max_sum, 1);
                pi_gpu.readN(&result, 1);

                EXPECT_THE_SAME(reference_max_sum, max_sum, "OpenCL1 result should be consistent!");
                EXPECT_THE_SAME(reference_result, (int) result, "OpenCL1 result should be consistent!");
                t.nextLap();
            }

            std::cout << "OpenCL #1: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "OpenCL #1: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                gpu::gpu_mem_32i gs_gpu[2];
                gpu::gpu_mem_32i ps_gpu[2];
                gpu::gpu_mem_32u pi_gpu[2];
                gs_gpu[0].resizeN(n);
                ps_gpu[0].resizeN(n);
                pi_gpu[0].resizeN(n);
                gs_gpu[0].writeN(as.data(), n);

                ocl::Kernel kernel_pref(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "max_prefix_sum_swap");
                kernel_pref.compile();

                unsigned int density = 1;
                int i = 0;
                for (unsigned int len = n; len > 1; len = ceil_div(len, WORK_GROUP_SIZE)) {
                    gpu::WorkSize workSize(WORK_GROUP_SIZE, len);
                    int j = (i + 1) % 2;
                    unsigned int len2 = ceil_div(len, WORK_GROUP_SIZE);
                    gs_gpu[j].resizeN(len2);
                    ps_gpu[j].resizeN(len2);
                    pi_gpu[j].resizeN(len2);
                    kernel_pref.exec(workSize, gs_gpu[i], ps_gpu[i], pi_gpu[i], gs_gpu[j], ps_gpu[j], pi_gpu[j], density, len);
                    density *= WORK_GROUP_SIZE;
                    i = j;
                }
                int max_sum = 0;
                unsigned int result = 0;
                ps_gpu[i].readN(&max_sum, 1);
                pi_gpu[i].readN(&result, 1);

                EXPECT_THE_SAME(reference_max_sum, max_sum, "OpenCL2 result should be consistent!");
                EXPECT_THE_SAME(reference_result, (int) result, "OpenCL2 result should be consistent!");
                t.nextLap();
            }

            std::cout << "OpenCL #2: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "OpenCL #2: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
    }
}
