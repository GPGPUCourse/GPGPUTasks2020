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
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();
    
    unsigned int workGroupSize = 256;
    
    int benchmarkingIters = 10;
    unsigned int log2_n = 24;
    unsigned int n = (1 << log2_n);
    
    std::vector<unsigned int> as(n, 0);
    FastRandom r(n);
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
    
    
    std::vector<unsigned int> gpu_sorted(n, 0);
    {
        std::vector<unsigned int> sums(2 * n - 1, 0);
        
        gpu::gpu_mem_32u sums_buf;
        gpu::gpu_mem_32u A1_buf;
        gpu::gpu_mem_32u A2_buf;
        sums_buf.resizeN(sums.size());
        A1_buf.resizeN(n);
        A2_buf.resizeN(n);
        
        std::string defines = "-DWORK_GROUP_SIZE=" + std::to_string(workGroupSize);
        ocl::Kernel sums_kern(radix_kernel, radix_kernel_length, "bit_sums", defines);
        ocl::Kernel radix_kern(radix_kernel, radix_kernel_length, "radix", defines);
        sums_kern.compile();
        radix_kern.compile();
    
        timer t;
    
        auto A1_ptr = &A1_buf;
        auto A2_ptr = &A2_buf;
        
        for (int iter = 0; iter < benchmarkingIters; iter++) {
            A1_ptr->writeN(as.data(), as.size());
            t.restart();
            
            for (unsigned int bit = 0; bit < 32; bit++) {
                for (int level = (int) log2_n; (1 << level) >= workGroupSize; level--) {
                    sums_kern.exec(gpu::WorkSize(workGroupSize, (1 << level)), sums_buf, *A1_ptr, log2_n, level, bit);
                }
                radix_kern.exec(gpu::WorkSize(workGroupSize, n), sums_buf, *A1_ptr, *A2_ptr, log2_n, bit);
                
                std::swap(A1_ptr, A2_ptr);
            }
            
            t.nextLap();
        }
        A1_ptr->readN(gpu_sorted.data(), n);
    
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;
    }
    
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(gpu_sorted[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }
    
    return 0;
}
