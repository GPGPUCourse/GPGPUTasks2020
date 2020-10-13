#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <numeric>

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

    for (int n = 32; n <= max_n; n *= 2) {
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
            gpu::Device device = gpu::chooseGPUDevice(argc, argv);

            gpu::Context context;
            context.init(device.device_id_opencl);
            context.activate();
            {
                // Create and compile kernel
                ocl::Kernel prefix_sum(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "max_prefix_sum");
                prefix_sum.compile();

                // Set-up NDRange
                const unsigned int workGroupSize = 128;

                // Create index vector
                std::vector<unsigned int> index_range(n);
                std::iota(index_range.begin(), index_range.end(), 1);

                // Set-up memory buffers
                gpu::gpu_mem_32i max_prefix_buffer; // Holds max prefixes
                max_prefix_buffer.resizeN(n);
                max_prefix_buffer.writeN(as.data(), n);

                gpu::gpu_mem_32i sum_buffer; // Holds sums (copy via VRAM to avoid PCI transfer)
                sum_buffer.resizeN(n);
                max_prefix_buffer.copyToN(sum_buffer, n);

                gpu::gpu_mem_32u index_buffer; // Holds max prefix indexes
                index_buffer.resizeN(n);
                index_buffer.writeN(index_range.data(), n);

                // Swap buffers
                gpu::gpu_mem_32i swap_max_prefix_buffer;
                swap_max_prefix_buffer.resizeN(n);

                gpu::gpu_mem_32i swap_sum_buffer;
                swap_sum_buffer.resizeN(n);

                gpu::gpu_mem_32u swap_index_buffer;
                swap_index_buffer.resizeN(n);

                // Reset buffers
                gpu::gpu_mem_32i gpu_reset_pref_buffer;
                gpu_reset_pref_buffer.resizeN(n);
                max_prefix_buffer.copyToN(gpu_reset_pref_buffer, n);

                gpu::gpu_mem_32i gpu_reset_sum_buffer;
                gpu_reset_sum_buffer.resizeN(n);
                sum_buffer.copyToN(gpu_reset_sum_buffer, n);

                gpu::gpu_mem_32u gpu_reset_idx_buffer;
                gpu_reset_idx_buffer.resizeN(n);
                index_buffer.copyToN(gpu_reset_idx_buffer, n);

                // Kernel runs
                timer t;
                for (int i = 0; i < benchmarkingIters; ++i) {

                    t.stop();
                    gpu_reset_pref_buffer.copyToN(max_prefix_buffer, n);
                    gpu_reset_sum_buffer.copyToN(sum_buffer, n);
                    gpu_reset_idx_buffer.copyToN(index_buffer, n);
                    t.start();

                    unsigned int remain_size = n;
                    unsigned int prev_remain_size;
                    while (true){
                        prev_remain_size = remain_size; // Save real remaining size
                        remain_size = ((remain_size + workGroupSize - 1) / workGroupSize) * workGroupSize; // Pad to match workGroupSize

                        prefix_sum.exec(gpu::WorkSize(workGroupSize, remain_size),
                                        max_prefix_buffer, sum_buffer, index_buffer,
                                        swap_max_prefix_buffer, swap_sum_buffer,
                                        swap_index_buffer, prev_remain_size);

                        remain_size /= workGroupSize; // Shrink NDRange

                        max_prefix_buffer.swap(swap_max_prefix_buffer);
                        sum_buffer.swap(swap_sum_buffer);
                        index_buffer.swap(swap_index_buffer);

                        if(remain_size == 1){
                            break;
                        }
                    }

                    int max_prefix_value = 0;
                    unsigned int max_pref_index = 0;
                    max_prefix_buffer.readN(&max_prefix_value, 1);
                    index_buffer.readN(&max_pref_index, 1);

                    EXPECT_THE_SAME(reference_max_sum, max_prefix_value, "GPU result should be consistent!");
                    EXPECT_THE_SAME((unsigned int)reference_result, max_pref_index, "GPU result should be consistent!");
                    t.nextLap();
                }

                std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
                std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
            }
        }
    }
}
