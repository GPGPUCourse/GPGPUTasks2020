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


int main(int argc, char **argv)
{
    bool DEBUG_PRINT = false;
    
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();
    
    unsigned int workGroupSize = 256;
    
    int benchmarkingIters = 10;
    unsigned int log2_n = 2;
    unsigned int n = workGroupSize;
    //unsigned int n = (1 << log2_n);
    
    std::vector<unsigned int> as(n, 0);
    FastRandom r(1564);
    for (unsigned int i = 0; i < n; ++i) {
        //as[i] =  r.next(0, 12);
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
        std::cout << "CPU: " << (n/1000.0/1000) / t.lapAvg() << " millions/s" << std::endl;
    }

    if (DEBUG_PRINT) {
        for (int i = 0; i < n; i++)
            std::cout << as[i] << ' ';
//        std::cout << "\nSorted:\n";
//        for (int i = 0; i < n; i++)
//            std::cout << cpu_sorted[i] << ' ';
        std::cout << '\n';
        std::cout << "GPU OUT:\n";
    }
    
/*    {
        std::vector<unsigned int> sums(2 * n - 1);
    
        gpu::gpu_mem_32u sums_buf;
        sums_buf.resizeN(2*n - 1);
        
        std::string defines = "-DWORK_GROUP_SIZE=" + std::to_string(workGroupSize);
        ocl::Kernel k_sums(radix_kernel, radix_kernel_length, "sums", defines);
        k_sums.compile();
        
        sums_buf.writeN(as.data(), n);
        
        for (unsigned int level = 1; level <= 1; level++) {
            k_sums.exec(gpu::WorkSize(workGroupSize, n), sums_buf, log2_n, level);
        }
        
        sums_buf.readN(sums.data(), 2 * n - 1);
        
        if (DEBUG_PRINT) {
            std::cout << "Result\n";
            for (int i = 0; i < 2*n-1; i++)
                std::cout << sums[i] << ' ';
            std::cout << "\n";
        }
    }*/
    {
        gpu::gpu_mem_32u as_gpu;
        gpu::gpu_mem_32u out_gpu;
        
        as_gpu.resizeN(n);
        out_gpu.resizeN(n);
        
        std::string defines = "-DWORK_GROUP_SIZE=" + std::to_string(workGroupSize);
        ocl::Kernel radix_kern(radix_kernel, radix_kernel_length, "local_radix", defines);
        radix_kern.compile();
        
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);
            t.restart();
            
            unsigned int global_work_size = n;

            radix_kern.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, out_gpu, n);

            t.nextLap();
        }
        
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n/1000.0/1000) / t.lapAvg() << " millions/s" << std::endl;
        
        out_gpu.readN(as.data(), n);
    }
    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }
    
    return 0;
}
