#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/merge_cl.h"

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


int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    unsigned int n = 32*1024*1024;
    std::vector<float> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.nextf();
    }
    /*
    for (int i = 0; i + 4 <= n; i += 4) {
        std::sort(as.begin() + i, as.begin() + i + 4);
        for (int j = i; j < i + 4; j++) std::cout << as[j] << " ";
        std::cout << "\n";
    }
    */
    /*
    for (int x : as) std::cout << x << " ";
    std::cout << "\n";
    */
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::vector<float> cpu_sorted;
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            cpu_sorted = as;
            std::sort(cpu_sorted.begin(), cpu_sorted.end());
            t.nextLap();
            //std::cout << "cpu iter = " << iter << "\n";
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;
    }

    gpu::gpu_mem_32f as_gpu;
    as_gpu.resizeN(n);
    gpu::gpu_mem_32f res_gpu;
    res_gpu.resizeN(n);
    {
        ocl::Kernel merge(merge_kernel, merge_kernel_length, "merge");
        merge.compile();
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);
            t.restart(); // Запускаем секундомер после прогрузки данных чтобы замерять время работы кернела, а не трансфер данных
            unsigned int workGroupSize = 128;
            unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
            int k = 1;
            while (k * 2 <= n) {
                //std::cout << "k = " << k << ", n = " << n << "\n";
                merge.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, res_gpu, n, k);
                /*
                std::vector<float> res(n);
                res_gpu.readN(res.data(), n);
                
                for (int i = 0; i + k * 2 <= n; i += k * 2) {
                    for (int j = i; j < i + k * 2; j++) std::cout << res[j] << " ";
                    std::cout << "\n";
                }
                std::cout << "\n";
                */
                k *= 2;
                std::swap(as_gpu, res_gpu);
            }
            t.nextLap();
            /**/
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;
        as_gpu.readN(as.data(), n);
        //std::cout << "readN+\n";
    }

    /*
    for (int i = 0; i < n; ++i) {
        std::cout << cpu_sorted[i] << " ";
    }
    std::cout << "\n";
    */
    // Проверяем корректность результатов
    
    for (int i = 0; i < n; ++i) {
        //if (i % int(3e6) == 0) std::cout << i << "\n";
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }
    
    return 0;
}
