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
        std::vector<unsigned int> indices(n, 0);
        FastRandom r(n);
        for (int i = 0; i < n; ++i) {
            as[i] = (unsigned int) r.next(-values_range, values_range);
            indices[i] = i;
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
        unsigned int old_n = n;
        {
            // TODO: implement on OpenCL
            gpu::Device device = gpu::chooseGPUDevice(argc, argv);
            gpu::Context context;
            context.init(device.device_id_opencl);
            context.activate();
            
            {
                ocl::Kernel kernel(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "max_prefix_sum");
                bool printLog = true;
                kernel.compile(printLog);

                timer t;
                
                gpu::gpu_mem_32i sum_vram[] = {
                    gpu::gpu_mem_32i::createN(n),
                    gpu::gpu_mem_32i::createN(n)
                };
                gpu::gpu_mem_32i pref_sum_vram[] = {
                    gpu::gpu_mem_32i::createN(n),
                    gpu::gpu_mem_32i::createN(n)
                };
                gpu::gpu_mem_32u pref_index_vram[] = {
                    gpu::gpu_mem_32u::createN(n),
                    gpu::gpu_mem_32u::createN(n)
                };
                
                for (int iter = 0; iter < benchmarkingIters; ++iter) {
                    n = old_n;
                    sum_vram[0].writeN(as.data(), n);
                    sum_vram[0].copyToN(pref_sum_vram[0], n);
                    pref_index_vram[0].writeN(indices.data(), n);
                
                    unsigned int work_group_size = 32;
                    unsigned int work_groups_number = (n + work_group_size - 1) / work_group_size;
                    unsigned int sum;
                    while (true) {
                        kernel.exec(gpu::WorkSize(work_group_size, work_groups_number * work_group_size),
                            sum_vram[0],
                            pref_sum_vram[0],
                            pref_index_vram[0],
                            ocl::LocalMem(work_group_size * sizeof(int)),
                            ocl::LocalMem(work_group_size * sizeof(int)),
                            ocl::LocalMem(work_group_size * sizeof(unsigned int)),
                            n,
                            sum_vram[1],
                            pref_sum_vram[1],
                            pref_index_vram[1]
                        );
                        if (work_groups_number == 1) {
                            break;
                        }
                        n = work_groups_number;
                        work_groups_number = (work_groups_number + work_group_size - 1) / work_group_size;
                        std::swap(sum_vram[0], sum_vram[1]);
                        std::swap(pref_sum_vram[0], pref_sum_vram[1]);
                        std::swap(pref_index_vram[0], pref_index_vram[1]);
                    }
                    int max_sum;
                    pref_sum_vram[1].readN(&max_sum, 1);
                    unsigned int result;
                    pref_index_vram[1].readN(&result, 1);
                    if (max_sum < 0) {
                        max_sum = 0;
                        result = 0;
                    } else {
                        result++;
                    }
                    EXPECT_THE_SAME(reference_max_sum, max_sum, "GPU result should be consistent!");
                    EXPECT_THE_SAME(reference_result, (int)result, "GPU result should be consistent!");
                    t.nextLap();
                }
                std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
                std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
            }
            n = old_n;
        }
    }
}
