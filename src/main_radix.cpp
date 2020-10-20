#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/radix_cl.h"

#include <vector>
#include <iostream>
#include <stdexcept>


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


void calc_prefix_bucket_sum(gpu::gpu_mem_32u bucket_sum, unsigned int bucket_cnt,
        ocl::Kernel bucket_cal_prefix_sum, ocl::Kernel recalc_prefix_sum) {
    int step = 1;
    for (; step < bucket_cnt; step *= 128) {
        unsigned int workGroupSize = 128;
        unsigned int cur_bucket_cnt = (bucket_cnt + step - 1) / step;
        unsigned int global_work_size = (cur_bucket_cnt + workGroupSize - 1) / workGroupSize * workGroupSize;
        bucket_cal_prefix_sum.exec(gpu::WorkSize(workGroupSize, global_work_size), bucket_sum, step, bucket_cnt);
    }

    step /= 128;
    for (; step >= 1; step /= 128) {
        unsigned int workGroupSize = 128;
        unsigned int cur_bucket_cnt = (bucket_cnt + step - 1) / step;
        unsigned int global_work_size = (cur_bucket_cnt + workGroupSize - 1) / workGroupSize * workGroupSize;
        recalc_prefix_sum.exec(gpu::WorkSize(workGroupSize, global_work_size), bucket_sum, step, bucket_cnt);
    }
}

void calc_prefix_sum(unsigned int workGroupSize, unsigned int global_work_size, gpu::gpu_mem_32u as_gpu,
        unsigned  int shift, unsigned  int n, gpu::gpu_mem_32u bucket_sum,
        gpu::gpu_mem_32u prefix_sum, ocl::Kernel bucket_cal_prefix_sum, ocl::Kernel recalc_prefix_sum, ocl::Kernel prefix_sum_calc) {
    /*std::vector<unsigned int> bsum((n + 127)/128 * 16, 0);
    int sum = 0;
    std::cout << "Before prefix sum:\n";
    bucket_sum.readN(bsum.data(), (n + 127)/128 * 16);
    for (int i = 0; i < 16; ++i) {
        std::cout << bsum[i] << " ";
        //sum += bsum[i * 16 + 1];
        //std::cout << "(" << sum << ")" << " ";
    }

    std::cout << "\n";*/

    calc_prefix_bucket_sum(bucket_sum, (n + 127)/128, bucket_cal_prefix_sum, recalc_prefix_sum);
    prefix_sum_calc.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, shift, n, bucket_sum, prefix_sum);

    /*std::cout << "After prefix sum:\n";
    bucket_sum.readN(bsum.data(), (n + 127)/128 * 16);
    for (int i = 0; i < (n + 127)/128; ++i) {
        std::cout << bsum[i] << " ";
    }

    std::cout << "\n"; */
}


int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    unsigned int n = 32 * 1024 * 1024;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<int>::max());
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::vector<unsigned int> cpu_sorted;
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            cpu_sorted = as;
            std::sort(cpu_sorted.begin(), cpu_sorted.end());
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;
    }

    gpu::gpu_mem_32u as_gpu, as_gpu_out, bucket_sum, prefix_sum, bucket_prefix_sum;
    as_gpu.resizeN(n);
    as_gpu_out.resizeN(n);
    prefix_sum.resizeN(n);
    bucket_sum.resizeN(((n + 127)/128) * 16);
    bucket_prefix_sum.resizeN(16);

    {
        ocl::Kernel radix_sum(radix_kernel, radix_kernel_length, "radix_sum");
        ocl::Kernel radix_exchange(radix_kernel, radix_kernel_length, "radix_exchange");
        ocl::Kernel bucket_cal_prefix_sum(radix_kernel, radix_kernel_length, "bucket_cal_prefix_sum");
        ocl::Kernel recalc_prefix_sum(radix_kernel, radix_kernel_length, "recalc_prefix_sum");
        ocl::Kernel prefix_sum_calc(radix_kernel, radix_kernel_length, "prefix_sum_calc");
        radix_sum.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart(); // Запускаем секундомер после прогрузки данных чтобы замерять время работы кернела, а не трансфер данных

            unsigned int workGroupSize = 128;
            unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;

            for (int shift = 0; shift < 32; shift += 4) {
                radix_sum.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, shift, n, bucket_sum);
                calc_prefix_sum(workGroupSize, global_work_size, as_gpu, shift, n, bucket_sum, prefix_sum, bucket_cal_prefix_sum, recalc_prefix_sum, prefix_sum_calc);
                radix_exchange.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, as_gpu_out, shift, n, prefix_sum, bucket_sum);
                std::swap(as_gpu, as_gpu_out);
            }

            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;

        as_gpu.readN(as.data(), n);
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}
