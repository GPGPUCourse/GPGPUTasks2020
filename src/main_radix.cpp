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
#include <cstring>


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

const unsigned int RadixCPUPow2 = 8;
const unsigned int RadixCPUPow2Value = 1 << RadixCPUPow2;
const unsigned int RadixCPUMask = RadixCPUPow2Value - 1;
const unsigned int RadixCPUNumOfIterations = (sizeof(unsigned int) * 8 + RadixCPUPow2 - 1) / RadixCPUPow2;

void RadixSortCPU( unsigned int *A, unsigned int N, unsigned int *B, bool *ResInA )
{
  unsigned int Count[RadixCPUPow2Value];
  unsigned int Offset[RadixCPUPow2Value];
  unsigned int *Src = A;
  unsigned int *Dst = B;

  for (unsigned int Power2 = 0, It = 0; It < RadixCPUNumOfIterations; Power2 += RadixCPUPow2, It++)
  {
    memset(Count, 0, sizeof(Count));

    for (unsigned int i = 0; i < N; i++)
      Count[(Src[i] >> Power2) & RadixCPUMask]++;

    Offset[0] = 0;
    for (unsigned int i = 1; i < RadixCPUPow2Value; i++)
      Offset[i] = Offset[i - 1] + Count[i - 1];
    
    for (unsigned int i = 0; i < N; i++)
      Dst[Offset[(Src[i] >> Power2) & RadixCPUMask]++] = Src[i];

    std::swap(Src, Dst);
  }

  *ResInA = RadixCPUNumOfIterations % 2 == 0;
}

int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    unsigned int n = 32 * 1024 * 1024;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<int>::max());
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::vector<unsigned int> cpu_sorted;
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            cpu_sorted = as;
            t.restart();
            std::sort(cpu_sorted.begin(), cpu_sorted.end());
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << (n/1000./1000) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        std::vector<unsigned int> RadixCpuA;
        std::vector<unsigned int> RadixCpuB(n);
        bool ResInA;
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            RadixCpuA = as;
            t.restart();
            RadixSortCPU(RadixCpuA.data(), n, RadixCpuB.data(), &ResInA);
            t.nextLap();
        }
        std::cout << "CPU Radix: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU Radix: " << (n/1000./1000) / t.lapAvg() << " millions/s" << std::endl;

        std::vector<unsigned int> &Res = ResInA ? RadixCpuA : RadixCpuB;

        for (int i = 0; i < n; ++i) {
          EXPECT_THE_SAME(cpu_sorted[i], Res[i], "CPU Radix results should be equal to CPU results!");
        }
    }

    gpu::gpu_mem_32u as_gpu;
    as_gpu.resizeN(n);
    gpu::gpu_mem_32u ResGPU;
    ResGPU.resizeN(n);

    {
        const unsigned int Pow2 = 4;
        const unsigned int Pow2Value = 1 << Pow2;
        const unsigned int Mask = Pow2Value - 1;
        const unsigned int NumOfIterations = (sizeof(unsigned int) * 8 + Pow2 - 1) / Pow2;
        
        const unsigned int WorkGroupPow2 = 6;
        const unsigned int WorkGroupSize = 1 << WorkGroupPow2;
        const unsigned int BlockSizeCD = 31;
        const unsigned int BlockSizeEO = 2;

        unsigned int CellSize = 1;

        std::string Defines = "-D POW2_VALUE=" + std::to_string(Pow2Value) + " ";
        Defines += "-D POW2=" + std::to_string(Pow2) + " ";
        Defines += "-D MASK=" + std::to_string(Mask) + " ";
        Defines += "-D GROUP_SIZE=" + std::to_string(WorkGroupSize) + " ";
        Defines += "-D GROUP_SIZE_POW2=" + std::to_string(WorkGroupPow2) + " ";
        Defines += "-D BLOCK_SIZE_CD=" + std::to_string(BlockSizeCD) + " ";
        Defines += "-D BLOCK_SIZE_EO=" + std::to_string(BlockSizeEO);

        ocl::Kernel CountDigits(radix_kernel, radix_kernel_length, "CountDigits", Defines);
        CountDigits.compile();
        ocl::Kernel EvalOffsets(radix_kernel, radix_kernel_length, "EvalOffsets", Defines);
        EvalOffsets.compile();
        ocl::Kernel Radix(radix_kernel, radix_kernel_length, "Radix", Defines);
        Radix.compile();

        gpu::gpu_mem_32u OffsetsGPU;
        gpu::gpu_mem_32u CountDigitsGPU;

        gpu::gpu_mem_32u *Src;
        gpu::gpu_mem_32u *Dst;

        std::vector<unsigned int> Offsets;
        Offsets.reserve(Pow2Value * (unsigned int)log2f(n) / Pow2 + Pow2Value * Pow2);

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart(); // Запускаем секундомер после прогрузки данных чтобы замерять время работы кернела, а не трансфер данных

            Offsets.clear();

            unsigned int SumArraySize = Pow2Value;
            unsigned int CurN = (n + BlockSizeCD - 1) / BlockSizeCD;
            unsigned int CurOffset = 0;

            while (CurN > 1)
            {
              SumArraySize += CurN * Pow2Value;

              for (unsigned int i = 0; i < Pow2Value; i++)
              {
                Offsets.push_back(CurOffset);
                CurOffset += CurN;
              }

              CurN = (CurN + BlockSizeEO - 1) / BlockSizeEO;
            }

            OffsetsGPU.resizeN(Offsets.size());
            OffsetsGPU.writeN(Offsets.data(), Offsets.size());
            CountDigitsGPU.resizeN(SumArraySize);

            Src = &as_gpu;
            Dst = &ResGPU;

            for (unsigned int Power2 = 0, It = 0; It < NumOfIterations; Power2 += Pow2, It++)
            {
              CurN = (n + BlockSizeCD - 1) / BlockSizeCD;

              unsigned int GlobalWorkSizeCD = (n + WorkGroupSize * BlockSizeCD - 1) / (WorkGroupSize * BlockSizeCD) * WorkGroupSize;
              CountDigits.exec(gpu::WorkSize(WorkGroupSize, GlobalWorkSizeCD),
                               *Src, CountDigitsGPU, n, CurN, Power2);

              CurOffset = 0;

              while (CurN > 1)
              {
                CurOffset += CurN * Pow2Value;

                unsigned int GlobalWorkSizeEO = (CurN + WorkGroupSize * BlockSizeEO - 1) / (WorkGroupSize * BlockSizeEO) * WorkGroupSize;

                EvalOffsets.exec(gpu::WorkSize(WorkGroupSize, GlobalWorkSizeEO),
                                 CountDigitsGPU, CurOffset - CurN * Pow2Value,
                                 CountDigitsGPU, CurOffset,
                                 (CurN + BlockSizeEO - 1) / BlockSizeEO, CurN);

                CurN = (CurN + BlockSizeEO - 1) / BlockSizeEO;
              }

              Radix.exec(gpu::WorkSize(WorkGroupSize, GlobalWorkSizeCD),
                         *Src, *Dst,
                         OffsetsGPU, CountDigitsGPU, n, Power2, CurOffset);

              std::swap(Src, Dst);
            }

            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n/1000./1000) / t.lapAvg() << " millions/s" << std::endl;

        Src->readN(as.data(), n);
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}
