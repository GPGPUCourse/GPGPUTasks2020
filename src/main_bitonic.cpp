#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/bitonic_cl.h"

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
    unsigned int n = 32 * 1024 * 1024;
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
        std::cout << "CPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;
    }

    gpu::gpu_mem_32f as_gpu;
    as_gpu.resizeN(n);

    {
      const unsigned int Pow2LocalSortSize = 10;
      const unsigned int Pow2LocalSortSizeValue = 1 << Pow2LocalSortSize;

      std::string Defines = "-D POW2_LOCAL_SORT=" + std::to_string(Pow2LocalSortSize) + " ";

      Defines += "-D POW2_LOCAL_SORT_VALUE=" + std::to_string(Pow2LocalSortSizeValue) + " ";

      ocl::Kernel BitonicLocalFirst(bitonic_kernel, bitonic_kernel_length, "BitonicLocalFirst", Defines);
      ocl::Kernel BitonicLocal(bitonic_kernel, bitonic_kernel_length, "BitonicLocal", Defines);
      ocl::Kernel BitonicFirst(bitonic_kernel, bitonic_kernel_length, "BitonicFirst", Defines);
      ocl::Kernel Bitonic(bitonic_kernel, bitonic_kernel_length, "Bitonic", Defines);
      
      BitonicLocalFirst.compile();
      BitonicLocal.compile();
      BitonicFirst.compile();
      BitonicFirst.compile();

      timer t;
      for (int iter = 0; iter < benchmarkingIters; ++iter) {
          as_gpu.writeN(as.data(), n);

          t.restart(); // Запускаем секундомер после прогрузки данных чтобы замерять время работы кернела, а не трансфер данных

          unsigned int WorkGroupSizeLocal = 64;
          unsigned int GlobalWorkSizeLocal = (n + Pow2LocalSortSizeValue - 1) / Pow2LocalSortSizeValue * WorkGroupSizeLocal;
          unsigned int WorkGroupSize = 64;
          unsigned int GlobalWorkSize = (n / 2 + WorkGroupSize - 1) / WorkGroupSize * WorkGroupSize;

          BitonicLocalFirst.exec(gpu::WorkSize(WorkGroupSizeLocal, GlobalWorkSizeLocal),
                                 as_gpu, n);

          for (unsigned int FirstStepSizeI = Pow2LocalSortSize; (1 << FirstStepSizeI) <= n; FirstStepSizeI++)
          {
            BitonicFirst.exec(gpu::WorkSize(WorkGroupSize, GlobalWorkSize),
                              as_gpu, (1 << FirstStepSizeI), n);

            for (int StepSizeI = FirstStepSizeI - 1; StepSizeI >= Pow2LocalSortSize; StepSizeI--)
            {
              Bitonic.exec(gpu::WorkSize(WorkGroupSize, GlobalWorkSize),
                           as_gpu, (1 << StepSizeI), n);
            }

            BitonicLocal.exec(gpu::WorkSize(WorkGroupSizeLocal, GlobalWorkSizeLocal),
                              as_gpu, n);
          }

          t.nextLap();
      }
      std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
      std::cout << "GPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;

      as_gpu.readN(as.data(), n);
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}
