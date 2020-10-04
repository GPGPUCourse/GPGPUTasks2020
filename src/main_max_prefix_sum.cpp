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

            unsigned int work_group_size = 128;
            unsigned int global_work_size = (n + work_group_size - 1) / work_group_size * work_group_size;

            gpu::gpu_mem_32i as_gpu;
            as_gpu.resizeN(n);
            as_gpu.writeN(as.data(), n);


            gpu::gpu_mem_32i as_gpu_copy;
            as_gpu_copy.resizeN(n);
            as_gpu.copyToN(as_gpu_copy, n);
            {
                ocl::Kernel kernel(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "max_prefix_sum");
                bool print_log{false};
                kernel.compile(print_log);
                timer t;
                for (int iter = 0; iter < benchmarkingIters; ++iter) {
                    int max_sum = 0;
                    int sum = 0;
                    int result = 0;
                    for (int i = global_work_size; i > 1; i /= 2)
                    {
                        gpu::gpu_mem_32i s_gpu_out;
                        s_gpu_out.resizeN(i / 2);
                        gpu::gpu_mem_32i p_gpu_out;
                        p_gpu_out.resizeN(i / 2);
                        kernel.exec(gpu::WorkSize(work_group_size, global_work_size),
                                    as_gpu, as_gpu_copy, i, s_gpu_out, p_gpu_out);

                        as_gpu.clmem();
                        as_gpu.resizeN(i / 2);
                        s_gpu_out.copyToN(as_gpu, i / 2);


                        as_gpu_copy.clmem();
                        as_gpu_copy.resizeN(i / 2);
                        p_gpu_out.copyToN(as_gpu_copy, i / 2);

//                        std::vector<int> tmp_p{i / 2};
//                        {
//                            p_gpu_out.readN(tmp_p.data(), i / 2);
//                        }

//                        std::vector<int> tmp_s{i / 2};
//                        s_gpu_out.readN(tmp_s.data(), i / 2);
                    }
                    as_gpu_copy.readN(&max_sum, 1);
                    if (max_sum < 0) {
                        max_sum = 0;
                    } else {
                        int sum = 0;
                        while(sum != max_sum) {
                            sum += as[++result - 1];
                        }
                    }

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
