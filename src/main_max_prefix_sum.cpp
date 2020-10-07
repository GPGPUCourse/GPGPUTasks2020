#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/max_prefix_sum_cl.h"

#ifdef __CLION_IDE__

#include "cl/max_prefix_sum.cl"

#endif


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);
    unsigned int workGroupSize = 128;
    int benchmarkingIters = 10;
    int maxN = (1 << 24);

    for (int n = 2; n <= maxN; n *= 2) {
        unsigned int globalWorkSize = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        std::cout << "______________________________________________" << std::endl;
        int valuesRange = std::min(1023, std::numeric_limits<int>::max() / n);
        std::cout << "n=" << n << " values in range: [" << (-valuesRange) << "; " << valuesRange << "]" << std::endl;
        std::vector<int> as(globalWorkSize, 0);
        int referenceMaxSum = std::numeric_limits<int>::min();
        unsigned int referenceResult = 0;
        {
            FastRandom r(n);
            int sum = 0;
            for (unsigned int i = 0; i < n; ++i) {
                as[i] = r.next(-valuesRange, valuesRange);
                sum += as[i];
                if (sum > referenceMaxSum) {
                    referenceMaxSum = sum;
                    referenceResult = i + 1;
                }
            }
            std::cout << "Max prefix sum: " << referenceMaxSum << " on prefix [0; " << referenceResult << ")"
                      << std::endl;
        }
        {
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                int maxSum = std::numeric_limits<int>::min();
                unsigned int result = 0;
                int sum = 0;
                for (unsigned int i = 0; i < n; ++i) {
                    sum += as[i];
                    if (sum > maxSum) {
                        maxSum = sum;
                        result = i + 1;
                    }
                }
                t.nextLap(false);
                EXPECT_THE_SAME(referenceMaxSum, maxSum, "CPU result should be consistent!");
                EXPECT_THE_SAME(referenceResult, result, "CPU result should be consistent!");
                t.restart();
            }
            std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "CPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            gpu::Context context;
            context.init(device.device_id_opencl);
            context.activate();
            ocl::Kernel maxPrefixSum(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "maxPrefixSum");
            maxPrefixSum.compile();
            auto sumGpu = gpu::gpu_mem_32i::createN(globalWorkSize);
            auto maxSumGpu = gpu::gpu_mem_32i::createN(globalWorkSize);
            auto resultGpu = gpu::gpu_mem_32u::createN(globalWorkSize);
            auto sumGpu2 = gpu::gpu_mem_32i::createN(globalWorkSize);
            auto maxSumGpu2 = gpu::gpu_mem_32i::createN(globalWorkSize);
            auto resultGpu2 = gpu::gpu_mem_32u::createN(globalWorkSize);
            auto sumGpu3 = gpu::gpu_mem_32i::createN(globalWorkSize);
            auto maxSumGpu3 = gpu::gpu_mem_32i::createN(globalWorkSize);
            auto resultGpu3 = gpu::gpu_mem_32u::createN(globalWorkSize);
            sumGpu.writeN(as.data(), n);
            maxSumGpu.writeN(as.data(), n);
            { // почему нет вариант с лямбдой?
                std::vector<unsigned int> result(globalWorkSize, 0);
                for (unsigned int i = 0; i != n; ++i) {
                    result[i] = i + 1;
                }
                resultGpu.writeN(result.data(), globalWorkSize);
            }
            sumGpu.copyToN(sumGpu2, globalWorkSize);
            maxSumGpu.copyToN(maxSumGpu2, globalWorkSize);
            resultGpu.copyToN(resultGpu2, globalWorkSize);
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                for (auto workSize = n; workSize > 1; workSize = (workSize + workGroupSize - 1) / workGroupSize) {
                    maxPrefixSum.exec(gpu::WorkSize(workGroupSize, workSize),
                                      sumGpu2, maxSumGpu2, resultGpu2,
                                      sumGpu3, maxSumGpu3, resultGpu3,
                                      workSize);
                    sumGpu2.swap(sumGpu3);
                    maxSumGpu2.swap(maxSumGpu3);
                    resultGpu2.swap(resultGpu3);
                }
                t.nextLap(false);
                int maxSum = std::numeric_limits<int>::min();
                maxSumGpu2.readN(&maxSum, 1);
                EXPECT_THE_SAME(referenceMaxSum, maxSum, "GPU result should be consistent!");
                unsigned int result = 0;
                resultGpu2.readN(&result, 1);
                EXPECT_THE_SAME(referenceResult, result, "GPU result should be consistent!");
                sumGpu.copyToN(sumGpu2, globalWorkSize);
                maxSumGpu.copyToN(maxSumGpu2, globalWorkSize);
                resultGpu.copyToN(resultGpu2, globalWorkSize);
                t.restart();
            }
            std::cout << "GPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU:     " << (globalWorkSize / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
    }
}
