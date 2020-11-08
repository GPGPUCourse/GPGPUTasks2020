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
    unsigned int workGroupSize = 256;
    unsigned int n = 256 * 128 * 1024;
    
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
        std::cout << "CPU: " << n / 1e6 / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        gpu::gpu_mem_32f as_gpu;
        gpu::gpu_mem_32f bs_gpu;
        
        as_gpu.resizeN(n);
        bs_gpu.resizeN(n);
        
        std::string defines = "-DWORK_GROUP_SIZE=" + std::to_string(workGroupSize);
        ocl::Kernel merge(merge_kernel, merge_kernel_length, "merge", defines);
        ocl::Kernel merge_local(merge_kernel, merge_kernel_length, "merge_local", defines);
        
        merge.compile();
        merge_local.compile();
        
        auto A_ptr = &as_gpu;
        auto B_ptr = &bs_gpu;
        
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            A_ptr->writeN(as.data(), n);
            
            t.restart();
    
            merge_local.exec(gpu::WorkSize(workGroupSize, n), *A_ptr);
            
            for (int k = workGroupSize * 2; k <= n; k*=2) {
                merge.exec(gpu::WorkSize(workGroupSize, n), *A_ptr, *B_ptr, n, k);
                std::swap(A_ptr, B_ptr);
            }
            
            t.nextLap();
        }
        
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << n / 1e6 / t.lapAvg() << " millions/s" << std::endl;
        
        A_ptr->readN(as.data(), n);
    }
    
    
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}
