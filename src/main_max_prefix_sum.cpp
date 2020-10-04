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
            gpu::Context context;
            context.init(device.device_id_opencl);
            context.activate();

            ocl::Kernel kernel(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "max_prefix_sum");
            kernel.compile();

            std::vector<int> idx(n, 0);
            for (int i = 0; i < n; ++i) {
                idx[i] = i + 1;
            }
            gpu::gpu_mem_32i sum, new_sum;
            gpu::gpu_mem_32i max_sum, new_max_sum;
            gpu::gpu_mem_32i index, new_index;

            sum.resizeN(n);
            sum.writeN(as.data(), n);
            max_sum.resizeN(n);
            max_sum.writeN(as.data(), n);
            index.resizeN(n);
            index.writeN(idx.data(), n);

            unsigned int work_group_size = 128;
            unsigned int work_size = n;

            // resize output data
            unsigned int out_size = (work_size + work_group_size - 1) / work_group_size;
            new_sum.resizeN(out_size);
            new_max_sum.resizeN(out_size);
            new_index.resizeN(out_size);

            timer t;
            for (int i = 0; i < benchmarkingIters; ++i) {
                do {
                    kernel.exec(gpu::WorkSize(work_group_size, work_size),
                                sum, max_sum, index,
                                work_size,
                                new_sum, new_max_sum, new_index);
                    work_size = (work_size + work_group_size - 1) / work_group_size;

                    // swap in and out
                    sum.swap(new_sum);
                    max_sum.swap(new_max_sum);
                    index.swap(new_index);
                    // we can skip resize and rewriting data
                    // cause we overwrite it in next iteration
                } while (work_size > 1);

                // read original gpu_data
                // after last swap it has output data
                int res_max_sum = 0;
                max_sum.readN(&res_max_sum, 1);

                int result = 0;
                index.readN(&result, 1);

                EXPECT_THE_SAME(reference_max_sum, res_max_sum, "GPU result should be consistent!");
                EXPECT_THE_SAME(reference_result, result, "GPU result should be consistent!");
                t.nextLap();
            }

            std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
    }
}
