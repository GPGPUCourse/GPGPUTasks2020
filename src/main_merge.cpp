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
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::vector<float> cpu_sorted;
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

    {
        unsigned int WG_SIZE=256;
        gpu::gpu_mem_32f tmp[2];
        tmp[0].resizeN(n);
        tmp[1].resizeN(n);
        ocl::Kernel merge(merge_kernel, merge_kernel_length, "merge");
        merge.compile();
        timer t;
        int i=0;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            tmp[0].writeN(as.data(), n);
            t.restart(); // Запускаем секундомер после прогрузки данных чтобы замерять время работы кернела, а не трансфер данных
            i=0;
            for(int step=1;step<n;step*=2,i++)
                merge.exec(gpu::WorkSize(WG_SIZE, n), tmp[i%2], tmp[1-i%2], n, step);
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;
        tmp[i%2].readN(as.data(), n);
    }
    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }
    
    /*{
        unsigned int WG_SIZE=256;
        unsigned int WG_COUNT=n/WG_SIZE;
        gpu::gpu_mem_32f tmp[2];
        tmp[0].resizeN(n);
        tmp[1].resizeN(n);
        gpu::gpu_mem_32u block_inds_x;
        gpu::gpu_mem_32u block_inds_y;
        block_inds_x.resizeN(WG_COUNT);
        block_inds_y.resizeN(WG_COUNT);
        
        ocl::Kernel merge_init(merge_kernel, merge_kernel_length, "merge_init");
        ocl::Kernel merge_local(merge_kernel, merge_kernel_length, "merge_local");
        ocl::Kernel merge_global(merge_kernel, merge_kernel_length, "merge_global");
        merge_init.compile();
        merge_local.compile();
        merge_global.compile();
        
        timer t;
        int i=0;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            tmp[0].writeN(as.data(), n);
            t.restart(); // Запускаем секундомер после прогрузки данных чтобы замерять время работы кернела, а не трансфер данных
            i=0;
            for(int step=1;step<WG_SIZE;step*=2,i++)
                merge_local.exec(gpu::WorkSize(WG_SIZE, n), tmp[i%2], tmp[1-i%2], n, step);
            for(int step=WG_SIZE;step<n;step*=2,i++)
            {
                merge_init.exec(gpu::WorkSize(WG_SIZE, WG_COUNT), tmp[i%2], block_inds_x, block_inds_y, WG_COUNT, step);
                merge_global.exec(gpu::WorkSize(WG_SIZE, n), tmp[i%2], tmp[1-i%2], block_inds_x, block_inds_y, n, step);
            }
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;
        tmp[i%2].readN(as.data(), n);
    }
    // Проверяем корректность результатов
    //for (int i = 0; i < n; ++i)
    //    std::cout<<as[i]<<"\n";
    //std::cout<<"----------------------------------------------\n";
    //for (int i = 0; i < n; ++i)
    //    std::cout<<cpu_sorted[i]<<"\n";
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }*/

    return 0;
}
