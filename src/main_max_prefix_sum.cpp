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

    for (int n = 1 << 10; n <= max_n; n *= 2) {
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

        gpu::Device device = gpu::chooseGPUDevice(argc, argv);
        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();
        {
            ocl::Kernel kernel(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "sum");

            bool printLog = false;
            kernel.compile(printLog);

            //device.printInfo();

            gpu::WorkSize ws(32, n);

            gpu::shared_device_buffer_typed<int> numbersBuffer;
            numbersBuffer.growN(*ws.clGlobalSize());

            as.resize(*ws.clGlobalSize());
            numbersBuffer.writeN(as.data(), as.size());

            gpu::shared_device_buffer_typed<int> resultsBuffer;
            gpu::shared_device_buffer_typed<int> maxPrefixPerGroupBuffer;
            gpu::shared_device_buffer_typed<int> maxPrefixIndexPerGroupBuffer;

            auto pointsCount = ceil(*ws.clGlobalSize() / *ws.clLocalSize());

            resultsBuffer.growN(pointsCount);
            maxPrefixPerGroupBuffer.growN(pointsCount);
            maxPrefixIndexPerGroupBuffer.growN(pointsCount);

            {
                timer t;
                for (int i = 0; i < benchmarkingIters; ++i) {

                    std::vector<int> accumulatedValues(pointsCount, 0);
                    std::vector<int> maxPrefixPerGroup(pointsCount, 0);
                    std::vector<int> maxPrefixIndexPerGroup(pointsCount, 0);

                    resultsBuffer.writeN(accumulatedValues.data(), accumulatedValues.size());

                    kernel.exec(ws,
                                numbersBuffer,
                                resultsBuffer,
                                maxPrefixPerGroupBuffer,
                                maxPrefixIndexPerGroupBuffer);

                    resultsBuffer.readN(accumulatedValues.data(), pointsCount);
                    maxPrefixPerGroupBuffer.readN(maxPrefixPerGroup.data(), pointsCount);
                    maxPrefixIndexPerGroupBuffer.readN(maxPrefixIndexPerGroup.data(), pointsCount);

                    int sum = 0;
                    int maxSum = 0;
                    int maxPrefixIndex = 0;
                    int blockIndex = 0;
                    for(int i = 0; i < accumulatedValues.size(); i++) {
                        int value = accumulatedValues[i];
                        sum += value;

                        int prefix = maxPrefixPerGroup[i];
                        if( (sum - value) + prefix > maxSum ) {
                            blockIndex = i;
                            maxSum = (sum - value) + prefix;
                            maxPrefixIndex = maxPrefixIndexPerGroup[blockIndex];
                        } else if ( sum > maxSum ) {
                            maxSum = sum;
                            blockIndex = i;
                        }
                    }

                    int currentBlockMaxSum = maxSum;
                    int currentBlockMaxSumIndex = maxPrefixIndexPerGroup[blockIndex];
                    if (blockIndex < maxPrefixPerGroup.size()) {

                        int blockMaxPrefix = maxPrefixPerGroup[blockIndex];
                        int blockSum = accumulatedValues[blockIndex];

                        if(blockMaxPrefix > blockSum) {
                            currentBlockMaxSum -= blockSum;
                            currentBlockMaxSum += blockMaxPrefix;
                            currentBlockMaxSumIndex = maxPrefixIndexPerGroup[blockIndex];
                        }
                    }

                    int nextBlockMaxSum = maxSum;
                    int nextBlockMaxSumIndex = maxPrefixIndexPerGroup[blockIndex];
                    if (blockIndex < maxPrefixPerGroup.size() - 1) {

                        int blockMaxPrefix = maxPrefixPerGroup[blockIndex + 1];
                        int blockSum = accumulatedValues[blockIndex + 1];

                        if(blockMaxPrefix > blockSum) {
                            nextBlockMaxSum += blockMaxPrefix;
                            nextBlockMaxSumIndex = maxPrefixIndexPerGroup[blockIndex + 1];
                        }
                    }

                    if (maxPrefixIndex == 0) {
                        if (currentBlockMaxSum > nextBlockMaxSum) {
                            maxSum = currentBlockMaxSum;
                            maxPrefixIndex = currentBlockMaxSumIndex;
                        } else {
                            maxSum = nextBlockMaxSum;
                            maxPrefixIndex = nextBlockMaxSumIndex;
                        }
                    }

                    maxPrefixIndex++;

                    EXPECT_THE_SAME(reference_result, maxPrefixIndex, "CPU GPU result should be consistent!");
                    EXPECT_THE_SAME(reference_max_sum, maxSum, "CPU GPU result should be consistent!");

                    t.nextLap();
                }

                std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
                std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
            }
        }
    }
}
