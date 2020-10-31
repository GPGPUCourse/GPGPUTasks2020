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
            as[i] = r.next(-values_range, values_range);
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
            // TODO: implement on OpenCL
            gpu::Device device = gpu::chooseGPUDevice(argc, argv);
            gpu::Context context;
            context.init(device.device_id_opencl);
            context.activate();

            unsigned int wg_size = 128;
            unsigned int global_work_size = (n + wg_size - 1) / wg_size * wg_size;

            gpu::gpu_mem_32i as_gpu;
            as_gpu.resizeN(n);
            as_gpu.writeN(as.data(), n);

            gpu::gpu_mem_32i ps_gpu;
            ps_gpu.resizeN(n);
            as_gpu.copyToN(ps_gpu, n);

            gpu::gpu_mem_32i as_gpu_origin;
            as_gpu_origin.resizeN(n);
            as_gpu.copyToN(as_gpu_origin, n);

            gpu::gpu_mem_32i ps_gpu_origin;
            ps_gpu_origin.resizeN(n);
            as_gpu.copyToN(ps_gpu_origin, n);

            std::vector<int> indexes(n);
            for (auto i = 0; i < n; ++i) {
                indexes[i] = i + 1;
            }
            gpu::gpu_mem_32i idx_gpu;
            idx_gpu.resizeN(n);
            idx_gpu.writeN(indexes.data(), n);

            gpu::gpu_mem_32i idx_gpu_origin;
            idx_gpu_origin.resizeN(n);
            idx_gpu.copyToN(idx_gpu_origin, n);

            {
                std::string defines = "-D WORK_GROUP_SIZE=" + std::to_string(wg_size);
                ocl::Kernel kernel(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "max_prefix_sum", defines);
                bool print_log{false};

                kernel.compile(print_log);
                timer t;

                for (int iter = 0; iter < benchmarkingIters; ++iter) {
                    int max_sum = 0;
                    int result = 0;

                    if (iter != 0) {
                        t.stop();  // there is really no need to take this time into account
                        as_gpu.resizeN(n);
                        as_gpu_origin.copyToN(as_gpu, n);
                        ps_gpu.resizeN(n);
                        ps_gpu_origin.copyToN(ps_gpu, n);
                        idx_gpu.resizeN(n);
                        idx_gpu_origin.copyToN(idx_gpu, n);
                        t.start();
                    }
                    gpu::gpu_mem_32i as_out;
                    gpu::gpu_mem_32i ps_out;
                    gpu::gpu_mem_32i idx_out;

                    for (unsigned int size = global_work_size; size > 1; size = (size + wg_size - 1) / wg_size * wg_size / wg_size) {
                        const auto out_size = size / wg_size;

                        as_out.resizeN(0 < out_size ? out_size : 1);
                        ps_out.resizeN(0 < out_size ? out_size : 1);
                        idx_out.resizeN(0 < out_size ? out_size : 1);

                        kernel.exec(gpu::WorkSize(wg_size, size), as_gpu, ps_gpu, idx_gpu, size == global_work_size ? n : size, as_out, ps_out, idx_out);

                        as_gpu.swap(as_out);
                        ps_gpu.swap(ps_out);
                        idx_gpu.swap(idx_out);
                    }

                    ps_gpu.readN(&max_sum, 1);
                    idx_gpu.readN(&result, 1);
                    EXPECT_THE_SAME(reference_max_sum, max_sum, "GPU result should be consistent!");
                    EXPECT_THE_SAME(reference_result, result, "GPU result should be consistent!");
                    t.nextLap();
                }
                std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
                std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
            }
        }
    }
}
