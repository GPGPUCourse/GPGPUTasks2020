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

    gpu::Device device = gpu::chooseGPUDevice(argc, argv);
    gpu::Context context;
      
    context.init(device.device_id_opencl);
    context.activate();

    ocl::Kernel kernel(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "MaxPrefixSum");
    ocl::Kernel KernelFirst(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "MaxPrefixSumFirst");

    KernelFirst.compile(false);
    kernel.compile(false);

    for (int n = 2; n <= max_n; n *= 2) {
        std::cout << "______________________________________________" << std::endl;
        int values_range = std::min(1023, std::numeric_limits<int>::max() / n);
        std::cout << "n=" << n << " values in range: [" << (-values_range) << "; " << values_range << "]" << std::endl;

        std::vector<int> as(n, 0);
        FastRandom r(n);
        for (int i = 0; i < n; ++i) {
            as[i] = r.next(-values_range, values_range);
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
          // TODO: implement on OpenCL

          gpu::gpu_mem_32i BuffersArray[5];
          gpu::gpu_mem_32u BuffersArrayInd[2];

          gpu::gpu_mem_32i
            *BufferSrcFirst{&BuffersArray[0]},
            *BufferSumSrc{&BuffersArray[1]},
            *BufferSumRes{&BuffersArray[2]},
            *BufferMaxSumSrc{&BuffersArray[3]},
            *BufferMaxSumRes{&BuffersArray[4]};

          gpu::gpu_mem_32u
            *BufferSrcInd{&BuffersArrayInd[0]},
            *BufferResInd{&BuffersArrayInd[1]};

          const unsigned int WorkGroupSize = 64;
          const unsigned int BlockSize = 64 + 1;  // +1 for solve bank conflicts
          const unsigned int GroupBlockSize = WorkGroupSize * BlockSize;

          const unsigned int OptimalCPU = GroupBlockSize;

          unsigned int CellSize = BlockSize;
          unsigned int CurN = n;
          unsigned int NumberOfGroup = (CurN + GroupBlockSize - 1) / GroupBlockSize;

          BufferSrcFirst->resizeN(n);
          BufferSrcFirst->writeN(as.data(), n);

          for (int i = 1; i < 5; i++)
            BuffersArray[i].resizeN(NumberOfGroup * WorkGroupSize);

          for (int i = 0; i < 2; i++)
            BuffersArrayInd[i].resizeN(NumberOfGroup * WorkGroupSize);

          std::vector<int> ResSumArrCPU(OptimalCPU);
          std::vector<int> ResMaxSumArrCPU(OptimalCPU);
          std::vector<unsigned int> ResMaxSumIndArrCPU(OptimalCPU);

          timer t;
          for (int iter = 0; iter < benchmarkingIters; ++iter)
          {
            int max_sum = 0;
            int sum = 0;
            int result = 0;

            CurN = n;
            NumberOfGroup = (CurN + GroupBlockSize - 1) / GroupBlockSize;
            CellSize = BlockSize;

            BufferSumSrc = &BuffersArray[1];
            BufferSumRes = &BuffersArray[2];
            BufferMaxSumSrc = &BuffersArray[3];
            BufferMaxSumRes = &BuffersArray[4];

            BufferSrcInd = &BuffersArrayInd[0];
            BufferResInd = &BuffersArrayInd[1];

            if (CurN > OptimalCPU)
            {
              KernelFirst.exec(gpu::WorkSize(WorkGroupSize, NumberOfGroup * WorkGroupSize),
                               *BufferSrcFirst, *BufferSumSrc, *BufferMaxSumSrc, *BufferSrcInd, CurN);

              CurN = NumberOfGroup * WorkGroupSize;
              NumberOfGroup = (CurN + GroupBlockSize - 1) / GroupBlockSize;

              while (CurN > OptimalCPU)
              {
                kernel.exec(gpu::WorkSize(WorkGroupSize, NumberOfGroup * WorkGroupSize),
                            *BufferSumSrc, *BufferMaxSumSrc, *BufferSrcInd,
                            *BufferSumRes, *BufferMaxSumRes, *BufferResInd, CurN, CellSize);

                std::swap(BufferSumSrc, BufferSumRes);
                std::swap(BufferMaxSumSrc, BufferMaxSumRes);
                std::swap(BufferSrcInd, BufferResInd);

                CellSize *= BlockSize;
                CurN = NumberOfGroup * WorkGroupSize;
                NumberOfGroup = (CurN + GroupBlockSize - 1) / GroupBlockSize;
              }

              BufferSumSrc->readN(ResSumArrCPU.data(), CurN);
              BufferMaxSumSrc->readN(ResMaxSumArrCPU.data(), CurN);
              BufferSrcInd->readN(ResMaxSumIndArrCPU.data(), CurN);

              for (int i = 0; i < CurN; ++i)
              {
                const int TryMaxSum = sum + ResMaxSumArrCPU[i];

                if (TryMaxSum > max_sum)
                {
                  max_sum = TryMaxSum;
                  result = i * CellSize + ResMaxSumIndArrCPU[i];
                }

                sum += ResSumArrCPU[i];
              }
            }
            else
            {
              for (int i = 0; i < n; ++i)
              {
                sum += as[i];
                if (sum > max_sum)
                {
                  max_sum = sum;
                  result = i + 1;
                }
              }
            }

            EXPECT_THE_SAME(reference_max_sum, max_sum, "CPU result should be consistent!");
            EXPECT_THE_SAME(reference_result, result, "CPU result should be consistent!");
            t.nextLap();
          }
          std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
          std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
    }
}
