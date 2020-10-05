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
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);
    
    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();
    
    const unsigned int workGroupSize = 256;
    std::string defines = "-DWORK_GROUP_SIZE=" + std::to_string(workGroupSize);
    ocl::Kernel kernel(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "max_prefix_sum", defines);
    kernel.compile(false);
    
    int benchmarkingIters = 10;
    int max_n = (1 << 24);

    for (int n = 2; n <= max_n; n *= 2) {
        std::cout << "______________________________________________" << std::endl;
        int values_range = std::min(1023, std::numeric_limits<int>::max() / n);
        std::cout << "n=" << n << " values in range: [" << (-values_range) << "; " << values_range << "]" << std::endl;

        std::vector<int> as(n, 0);
        FastRandom r(n);
        for (int i = 0; i < n; ++i) {
            as[i] = (unsigned int) r.next(-values_range, values_range); // ?? confusing cast from (int) to (unsigned int) and then back to (int)
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

        {
            //unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
            unsigned int global_work_size = n;
    
            int out_sum;
            int out_prefix;
            int out_pid;
            
            gpu::gpu_mem_32i in_sum_buf, in_prefix_buf, in_pid_buf;
            gpu::gpu_mem_32i out_sum_buf, out_prefix_buf, out_pid_buf;
            
            in_sum_buf.resizeN(global_work_size);
            in_prefix_buf.resizeN(global_work_size);
            in_pid_buf.resizeN(global_work_size);
            
            out_sum_buf.resizeN((global_work_size + workGroupSize - 1) / workGroupSize);
            out_prefix_buf.resizeN((global_work_size + workGroupSize - 1) / workGroupSize);
            out_pid_buf.resizeN((global_work_size + workGroupSize - 1) / workGroupSize);
    
            std::vector<int> range(global_work_size);
            for (int i = 0; i < global_work_size; i++) range[i] = i + 1;
    
            if (benchmarkingIters == 1) {
                in_sum_buf.writeN(as.data(), global_work_size);
                in_prefix_buf.writeN(as.data(), global_work_size);
                in_pid_buf.writeN(range.data(), global_work_size);
            }
            
            timer t;
            for (int iter = 0; iter < benchmarkingIters; iter++) {
                if (benchmarkingIters != 1) {
                    in_sum_buf.writeN(as.data(), global_work_size);
                    in_prefix_buf.writeN(as.data(), global_work_size);
                    in_pid_buf.writeN(range.data(), global_work_size);
                }
                
                gpu::gpu_mem_32i * p_in_sum = &in_sum_buf;
                gpu::gpu_mem_32i * p_out_sum = &out_sum_buf;
                
                gpu::gpu_mem_32i * p_in_pref = &in_prefix_buf;
                gpu::gpu_mem_32i * p_out_pref = &out_prefix_buf;
                
                gpu::gpu_mem_32i * p_in_pid = &in_pid_buf;
                gpu::gpu_mem_32i * p_out_pid = &out_pid_buf;
                
                for (unsigned int i = global_work_size; i > 1; i = (i + workGroupSize - 1) / workGroupSize)
                {
                    kernel.exec(gpu::WorkSize(workGroupSize, i),
                                *p_in_sum, *p_in_pref, *p_in_pid,
                                *p_out_sum, *p_out_pref, *p_out_pid,
                                i);
                    
                    unsigned int out_size = (i + workGroupSize - 1) / workGroupSize;
                    
                    std::swap(p_in_sum, p_out_sum);
                    std::swap(p_in_pref, p_out_pref);
                    std::swap(p_in_pid, p_out_pid);
                }
                
                p_in_pid->readN(&out_pid, 1);
                p_in_pref->readN(&out_prefix, 1);
                
                EXPECT_THE_SAME(reference_max_sum, out_prefix, "GPU result should be consistent!");
                EXPECT_THE_SAME(reference_result, out_pid, "GPU result should be consistent!");
                t.nextLap();
            }
    
            std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
    }
}
