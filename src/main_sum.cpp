#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/sum_cl.h"

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

    unsigned int reference_sum = 0;
    unsigned int n = 10*1000*1000;///100*1000*1000;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(42);
    for (int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<unsigned int>::max() / n);
        reference_sum += as[i];
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            #pragma omp parallel for reduction(+:sum)
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU OpenMP result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU OMP: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU OMP: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    // TODO: implement on OpenCL
    gpu::Device device = gpu::chooseGPUDevice(argc,argv);
    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();
    
    const unsigned int wg_size=128;
    unsigned int global_size=(n+wg_size-1)/wg_size*wg_size;
    
    gpu::gpu_mem_32u as_gpu,sum_gpu;
    as_gpu.resizeN(n);
    as_gpu.writeN(as.data(),n);
    sum_gpu.resizeN(1);
    
    {
        const std::string kernel_name="sum_1";
        ocl::Kernel kernel(sum_kernel,sum_kernel_length,kernel_name);
        kernel.compile();
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum=0;
            sum_gpu.writeN(&sum,1);
            kernel.exec(gpu::WorkSize(wg_size,global_size),as_gpu,n,sum_gpu);
            sum_gpu.readN(&sum,1);
            EXPECT_THE_SAME(reference_sum, sum, "GPU " + kernel_name + " result should be consistent!");
            t.nextLap();
        }
        std::cout << "GPU " + kernel_name + ": " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU " + kernel_name + ": " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }
    
    {
        gpu::gpu_mem_32u as_gpu;
        as_gpu.resizeN(global_size);
        as_gpu.writeN(as.data(),n);
        const std::string kernel_name="sum_1_1";
        ocl::Kernel kernel(sum_kernel,sum_kernel_length,kernel_name);
        kernel.compile();
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum=0;
            sum_gpu.writeN(&sum,1);
            kernel.exec(gpu::WorkSize(wg_size,global_size),as_gpu,global_size,sum_gpu);
            sum_gpu.readN(&sum,1);
            EXPECT_THE_SAME(reference_sum, sum, "GPU " + kernel_name + " result should be consistent!");
            t.nextLap();
        }
        std::cout << "GPU " + kernel_name + ": " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU " + kernel_name + ": " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }
    
    {
        const std::string kernel_name="sum_2";
        ocl::Kernel kernel(sum_kernel,sum_kernel_length,kernel_name);
        kernel.compile();
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum=0;
            sum_gpu.writeN(&sum,1);
            kernel.exec(gpu::WorkSize(wg_size,global_size),as_gpu,n,sum_gpu);
            sum_gpu.readN(&sum,1);
            EXPECT_THE_SAME(reference_sum, sum, "GPU " + kernel_name + " result should be consistent!");
            t.nextLap();
        }
        std::cout << "GPU " + kernel_name + ": " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU " + kernel_name + ": " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }
    
    {
        const std::string kernel_name="sum_2_1";
        ocl::Kernel kernel(sum_kernel,sum_kernel_length,kernel_name);
        kernel.compile();
        gpu::gpu_mem_32u tmp_gpu[2];
        tmp_gpu[0].resizeN(n);
        tmp_gpu[1].resizeN(n);
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            tmp_gpu[0].writeN(as.data(),n);
            unsigned int sum=0,i=0,size=n;
            
            for(;size>1;size=(size+wg_size-1)/wg_size,i++)
                kernel.exec(gpu::WorkSize(wg_size,size),tmp_gpu[i%2],size,tmp_gpu[1-i%2]);
            
            tmp_gpu[i%2].readN(&sum,1);
            EXPECT_THE_SAME(reference_sum, sum, "GPU " + kernel_name + " result should be consistent!");
            t.nextLap();
        }
        std::cout << "GPU " + kernel_name + ": " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU " + kernel_name + ": " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        
    }
    
    /*{
        const std::string kernel_name="sum_2_2";
        ocl::Kernel kernel(sum_kernel,sum_kernel_length,kernel_name);
        kernel.compile();
        gpu::gpu_mem_32u tmp_gpu[2];
        tmp_gpu[0].resizeN(n);
        tmp_gpu[1].resizeN(n);
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            tmp_gpu[0].writeN(as.data(),n);
            unsigned int sum=0,i=0,size=n;
            
            for(;size>1;size=(size+wg_size-1)/wg_size,i++)
                kernel.exec(gpu::WorkSize(wg_size,size),tmp_gpu[i%2],size,tmp_gpu[1-i%2]);
            
            tmp_gpu[i%2].readN(&sum,1);
            EXPECT_THE_SAME(reference_sum, sum, "GPU " + kernel_name + " result should be consistent!");
            t.nextLap();
        }
        std::cout << "GPU " + kernel_name + ": " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU " + kernel_name + ": " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        
    }*/
    
    return 0;
}
