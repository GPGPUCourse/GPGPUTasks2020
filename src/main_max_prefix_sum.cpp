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
        FastRandom r(n);
        for (int i = 0; i < n; ++i) {
            as[i] = (unsigned int) r.next(-values_range, values_range);
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

        // TODO: implement on OpenCL
        gpu::Device device = gpu::chooseGPUDevice(argc,argv);
        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();
        
        const unsigned int wg_size=128;
        unsigned int global_size=(n+wg_size-1)/wg_size*wg_size;
        
        ocl::Kernel init(max_prefix_sum_kernel,max_prefix_sum_kernel_length,"init");
        init.compile();
        
        gpu::gpu_mem_32i as_gpu;
        as_gpu.resizeN(n);
        as_gpu.writeN(as.data(),n);
        
        gpu::gpu_mem_32i sum_gpu[2],result_gpu[2],max_sum_gpu[2];
        sum_gpu[0].resizeN(n);sum_gpu[1].resizeN(n);
        result_gpu[0].resizeN(n);result_gpu[1].resizeN(n);
        max_sum_gpu[0].resizeN(n);max_sum_gpu[1].resizeN(n);
        
        {
            const std::string kernel_name="max_prefix_sum";
            ocl::Kernel kernel(max_prefix_sum_kernel,max_prefix_sum_kernel_length,kernel_name);
            kernel.compile();
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                init.exec(gpu::WorkSize(wg_size,n),as_gpu,n,sum_gpu[0],result_gpu[0],max_sum_gpu[0]);
                unsigned int i=0,size=n;
                int max_sum=0,result=0;
            
                for(;size>1;size=(size+wg_size-1)/wg_size,i++)
                {
                    kernel.exec(gpu::WorkSize(wg_size,size),
                                sum_gpu[i%2],result_gpu[i%2],max_sum_gpu[i%2],
                                size,
                                sum_gpu[1-i%2],result_gpu[1-i%2],max_sum_gpu[1-i%2]);
                }
                
                max_sum_gpu[i%2].readN(&max_sum,1);
                result_gpu[i%2].readN(&result,1);
                if(max_sum<0)
                {
                    max_sum=0;
                    result=0;
                }
                EXPECT_THE_SAME(reference_max_sum, max_sum, "CPU result should be consistent!");
                EXPECT_THE_SAME(reference_result, result, "CPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
        
        {
            const std::string kernel_name="max_prefix_sum_2";
            ocl::Kernel kernel(max_prefix_sum_kernel,max_prefix_sum_kernel_length,kernel_name);
            kernel.compile();
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                init.exec(gpu::WorkSize(wg_size,n),as_gpu,n,sum_gpu[0],result_gpu[0],max_sum_gpu[0]);
                unsigned int i=0,size=n;
                int max_sum=0,result=0;
            
                for(;size>1;size=(size+wg_size-1)/wg_size,i++)
                {
                    kernel.exec(gpu::WorkSize(wg_size,size),
                                sum_gpu[i%2],result_gpu[i%2],max_sum_gpu[i%2],
                                size,
                                sum_gpu[1-i%2],result_gpu[1-i%2],max_sum_gpu[1-i%2]);
                }
                
                max_sum_gpu[i%2].readN(&max_sum,1);
                result_gpu[i%2].readN(&result,1);
                if(max_sum<0)
                {
                    max_sum=0;
                    result=0;
                }
                EXPECT_THE_SAME(reference_max_sum, max_sum, "CPU result should be consistent!");
                EXPECT_THE_SAME(reference_result, result, "CPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU _2: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU _2: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
    }
}
