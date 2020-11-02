#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/prefix_sum_cl.h"

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
    int benchmarkingIters = 1;
    int max_n = 10;(1 << 24);

    for (int n = 10; n <= max_n; n *= 2) {
        std::cout << "______________________________________________" << std::endl;
        int values_range = std::min(1023, std::numeric_limits<int>::max() / n);
        std::cout << "n=" << n << " values in range: [" << (-values_range) << "; " << values_range << "]" << std::endl;

        std::vector<int> as(n, 0);
        FastRandom r(n);
        for (int i = 0; i < n; ++i) {
            as[i] = r.next(-values_range, values_range);
            std::cout << as[i] << ' ';
        }
        std::cout << '\n';
        std::vector<int> reference_sum(n, 0);
        {
            int sum = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
                reference_sum[i] = sum;
            }
        }
        std::cout << "Prefix sum: \n";
        for (int i = 0; i < n; ++i) {
            std::cout << reference_sum[i] << ' ';
        }
        std::cout << '\n';

        {
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                int sum = 0;
                for (int i = 0; i < n; ++i) {
                    sum += as[i];
                    EXPECT_THE_SAME(reference_sum[i], sum, "CPU result should be consistent!");
                }
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

            gpu::gpu_mem_32i as_gpu_origin;
            as_gpu_origin.resizeN(n);
            as_gpu.copyToN(as_gpu_origin, n);

            std::vector<int> sums(n, 0);
            gpu::gpu_mem_32i sums_gpu;
            sums_gpu.resizeN(n);
            sums_gpu.writeN(sums.data(), n);

            gpu::gpu_mem_32i sums_gpu_origin;
            sums_gpu_origin.resizeN(n);
            sums_gpu.copyToN(sums_gpu_origin, n);

            {
                std::string defines = "-D WORK_GROUP_SIZE=" + std::to_string(wg_size);
                ocl::Kernel kernel(prefix_sum, prefix_sum_length, "prefix_sum", defines);
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
                        sums_gpu.resizeN(n);
                        sums_gpu_origin.copyToN(sums_gpu, n);
                        t.start();
                    }
                    gpu::gpu_mem_32i as_out;

                    for (unsigned int size = global_work_size, pow = 0; size > 1; size = (size + wg_size - 1) / wg_size * wg_size / wg_size) {
                        const auto out_size = size / wg_size;
                        as_out.resizeN(0 < out_size ? out_size : 1);
                        kernel.exec(gpu::WorkSize(wg_size, size), as_gpu, sums_gpu, n, size == global_work_size ? n : size, as_out, pow);

                        as_gpu.swap(as_out);
                    }

//                    ps_gpu.readN(&max_sum, 1);
//                    sums_gpu.readN(&result, 1);
//                    EXPECT_THE_SAME(reference_max_sum, max_sum, "GPU result should be consistent!");
//                    EXPECT_THE_SAME(reference_result, result, "GPU result should be consistent!");
                    t.nextLap();
                }
//                std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
//                std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
            }
        }
    }
}
