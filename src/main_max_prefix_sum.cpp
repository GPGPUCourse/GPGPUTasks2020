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

void calc_prefix_sum(std::vector<int>& a, unsigned int workGroupSize, ocl::Kernel& kernel_sum_in_bucket, ocl::Kernel& kernel_calc_prefix_sum) {
    if (a.size() < 256*workGroupSize) {
        for (int i = 1; i < a.size(); ++i) {
            a[i] += a[i - 1];
        }
    } else {
        unsigned int global_work_size = (a.size() + workGroupSize - 1) / workGroupSize * workGroupSize;
        std::vector<int> bucket_prefix_sum(a.size()/workGroupSize + 2, 0);
        gpu::gpu_mem_32i as_gpu;
        gpu::gpu_mem_32i bucket_sum;
        gpu::gpu_mem_32i sum;
        unsigned int n = a.size();
        as_gpu.resizeN(n);
        sum.resizeN(n);
        bucket_sum.resizeN(n/workGroupSize + 1);

        as_gpu.writeN(a.data(), n);

        kernel_sum_in_bucket.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, bucket_sum, n);

        bucket_sum.readN(bucket_prefix_sum.data() + 1, a.size()/workGroupSize + 1);
        calc_prefix_sum(bucket_prefix_sum, workGroupSize, kernel_sum_in_bucket, kernel_calc_prefix_sum);
        bucket_sum.writeN(bucket_prefix_sum.data(), n/workGroupSize + 1);

        kernel_calc_prefix_sum.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, bucket_sum, sum, n);

        sum.readN(a.data(), n);
    }
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
            gpu::Device device = gpu::chooseGPUDevice(argc, argv);

            gpu::Context context;
            context.init(device.device_id_opencl);
            context.activate();

            ocl::Kernel kernel_max_prefix_sum(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "max_prefix_sum");
            ocl::Kernel kernel_sum_in_bucket(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "sum_in_bucket");
            ocl::Kernel kernel_calc_prefix_sum(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "calc_prefix_sum");

            bool printLog = false;
            kernel_max_prefix_sum.compile(printLog);
            kernel_sum_in_bucket.compile(printLog);
            kernel_calc_prefix_sum.compile(printLog);

            int gpu_sum, gpu_res;
            unsigned int workGroupSize = 128;
            unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;

            std::vector<int> bucket_prefix_sum(n/workGroupSize + 2, 0);

            gpu::gpu_mem_32i as_gpu;
            gpu::gpu_mem_32i bucket_sum;
            gpu::gpu_mem_32i max_sum_gpu;
            gpu::gpu_mem_32i res_gpu;
            as_gpu.resizeN(n);
            bucket_sum.resizeN(n/workGroupSize + 1);
            max_sum_gpu.resizeN(1);
            res_gpu.resizeN(1);

            as_gpu.writeN(as.data(), n);
            bucket_sum.writeN(bucket_prefix_sum.data(), bucket_prefix_sum.size() - 1);

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                gpu_sum = -100000000;
                max_sum_gpu.writeN(&gpu_sum, 1);
                res_gpu.writeN(&gpu_sum, 1);

                kernel_sum_in_bucket.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, bucket_sum, n);

                bucket_sum.readN(bucket_prefix_sum.data() + 1, n/workGroupSize + 1);
                calc_prefix_sum(bucket_prefix_sum, workGroupSize, kernel_sum_in_bucket, kernel_calc_prefix_sum);
                bucket_sum.writeN(bucket_prefix_sum.data(), n/workGroupSize + 1);

                kernel_max_prefix_sum.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, bucket_sum, max_sum_gpu, res_gpu, n);

                res_gpu.readN(&gpu_res, 1);
                max_sum_gpu.readN(&gpu_sum, 1);

                EXPECT_THE_SAME(reference_max_sum, gpu_sum, "GPU result should be consistent!");
                EXPECT_THE_SAME(reference_result, gpu_res, "GPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
    }
}
