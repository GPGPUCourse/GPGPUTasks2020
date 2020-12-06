#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/max_prefix_sum_cl.h"

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


int main(int argc, char **argv) {
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
        std::cout << "Max prefix sum: " << reference_max_sum << " on prefix [0; " << reference_result << ")"
                  << std::endl;

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
            // chooseGPUDevice:
            // - Если не доступо ни одного устройства - кинет ошибку
            // - Если доступно ровно одно устройство - вернет это устройство
            // - Если доступно N>1 устройства:
            //   - Если аргументов запуска нет или переданное число не находится в диапазоне от 0 до N-1 - кинет ошибку
            //   - Если аргумент запуска есть и он от 0 до N-1 - вернет устройство под указанным номером
            gpu::Device device = gpu::chooseGPUDevice(argc, argv);

            // Этот контекст после активации будет прозрачно использоваться при всех вызовах в libgpu библиотеке
            // это достигается использованием thread-local переменных, т.е. на самом деле контекст будет активирован для текущего потока исполнения
            gpu::Context context;
            context.init(device.device_id_opencl);
            context.activate();
            ocl::Kernel kernel(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "max_prefix_sum");
            kernel.compile(false);
            gpu::gpu_mem_32i gpu_as, gpu_ps, gpu_out_as, gpu_out_ps;
            gpu::gpu_mem_32u gpu_results, gpu_out_results;

            gpu_as.resizeN(n);
            gpu_ps.resizeN(n);
            gpu_results.resizeN(n);
            gpu_out_as.resizeN(n);
            gpu_out_ps.resizeN(n);
            gpu_out_results.resizeN(n);

            std::vector<int> zeros(n, 0);
            std::vector<unsigned int> indexes(n);
            for (unsigned int i = 0; i < n; ++i) {
                indexes[i] = i;
            }

            const unsigned int workGroupSize = 256;

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                gpu_as.writeN(as.data(), n);
                gpu_ps.writeN(zeros.data(), n);
                gpu_results.writeN(indexes.data(), n);
                unsigned int curn = n;
                while (curn >= workGroupSize) {
                    kernel.exec(
                            gpu::WorkSize(workGroupSize, (curn + workGroupSize - 1) / workGroupSize * workGroupSize),
                            gpu_as, gpu_ps, gpu_results, gpu_out_as, gpu_out_ps, gpu_out_results, curn);
                    gpu_as.swap(gpu_out_as);
                    gpu_ps.swap(gpu_out_ps);
                    gpu_results.swap(gpu_out_results);
                    curn = (curn + workGroupSize - 1) / workGroupSize;
                }
                std::vector<int> sums(curn), psums(curn);
                std::vector<unsigned int> results(curn);

                gpu_as.readN(sums.data(), curn);
                gpu_ps.readN(psums.data(), curn);
                gpu_results.readN(results.data(), curn);

                int max_sum = 0, cur_sum = 0;
                int result = 0;
                for (unsigned int i = 0; i < curn; ++i) {
                    if (psums[i] + cur_sum > max_sum) {
                        max_sum = psums[i] + cur_sum;
                        result = results[i];
                    }
                    cur_sum += sums[i];
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
