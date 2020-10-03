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

struct Block{
    unsigned int SumBlock;
    unsigned int MaxElem;
    unsigned int Index;
};

int main(int argc, char **argv)
{
    int benchmarkingIters = 10;
    int max_n = (1 << 24);

    for (int n = 2; n <= max_n; n *= 2) {
    // for (int n = (1 << 24); n <= (1 << 24); n *= 2) {
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

        {
            gpu::Device device = gpu::chooseGPUDevice(argc, argv);
            gpu::Context context;
            context.init(device.device_id_opencl);
            context.activate();

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                gpu::gpu_mem_32i blocks[2], maxSum[2], index[2];
                blocks[0].resizeN(n), blocks[1].resizeN(n);
                maxSum[0].resizeN(n), maxSum[1].resizeN(n);
                index[0].resizeN(n), index[1].resizeN(n);
                blocks[0].writeN(as.data(), n);

                ocl::Kernel max_prefix_sum(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "max_prefix_sum");
                max_prefix_sum.compile();

                unsigned int curSize = n, curIter = 0;

                unsigned int groupSize = 128;
                unsigned int rangePerWorkItem = 2;

                unsigned int blockSize = 1;
                while (curSize >= groupSize * rangePerWorkItem) {
                    unsigned int workSize = ((curSize - 1) / groupSize + 1) * groupSize / rangePerWorkItem;
                    unsigned int id0 = curIter & 1, id1 = (curIter & 1) ^ 1;
                    max_prefix_sum.exec(gpu::WorkSize(groupSize, workSize),
                                        blocks[id0], maxSum[id0], index[id0], 
                                        blocks[id1], maxSum[id1], index[id1],
                                        blockSize, curIter == 0 ? 1 : 0, curSize);
                    curSize = (curSize - 1) / rangePerWorkItem;
                    curIter++;
                    blockSize *= rangePerWorkItem;
                }

                std::vector<int> resBlocks(curSize), resMaxSum(curSize), resIndex(curSize);
                blocks[curIter & 1].readN(resBlocks.data(), curSize);
                maxSum[curIter & 1].readN(resMaxSum.data(), curSize);
                index[curIter & 1].readN(resIndex.data(), curSize);

                if (curIter == 0){
                    for (int i = 0; i < curSize; i++){
                        resMaxSum[i] = std::max(0, resBlocks[i]);
                        resIndex[i] = resBlocks[i] > 0 ? 1 : 0;
                    }
                }

                int curSum = 0, ansSum = 0, ansIndex = 0;
                for (int i = 0; i < curSize; ++i) {
                    if (curSum + resMaxSum[i] > ansSum) {
                        ansSum = curSum + resMaxSum[i];
                        ansIndex = i * blockSize + resIndex[i];
                    }
                    curSum += resBlocks[i];
                }


                EXPECT_THE_SAME(reference_max_sum, ansSum, "GPU result should be consistent!");
                EXPECT_THE_SAME(reference_result, ansIndex, "GPU result should be consistent!");
                t.nextLap();
            }

            std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
    }
}
