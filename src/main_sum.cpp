#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/sum_cl.h"

template <typename T>
void raiseFail(const T &a, const T &b, std::string message,
               std::string filename, int line) {
  if (a != b) {
    std::cerr << message << " But " << a << " != " << b << ", " << filename
              << ":" << line << std::endl;
    throw std::runtime_error(message);
  }
}

#define EXPECT_THE_SAME(a, b, message)                                         \
  raiseFail(a, b, message, __FILE__, __LINE__)

int main(int argc, char **argv) {
  int benchmarkingIters = 10;

  unsigned int reference_sum = 0;
  unsigned int n = 100 * 1000 * 1000;
  std::vector<unsigned int> as(n, 0);
  FastRandom r(42);
  for (int i = 0; i < n; ++i) {
    as[i] = (unsigned int)r.next(std::numeric_limits<unsigned int>::max() / n);
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
    std::cout << "CPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s"
              << std::endl;
    std::cout << "CPU:     " << (n / 1000.0 / 1000.0) / t.lapAvg()
              << " millions/s" << std::endl;
  }

  {
    timer t;
    for (int iter = 0; iter < benchmarkingIters; ++iter) {
      unsigned int sum = 0;
#pragma omp parallel for reduction(+ : sum)
      for (int i = 0; i < n; ++i) {
        sum += as[i];
      }
      EXPECT_THE_SAME(reference_sum, sum,
                      "CPU OpenMP result should be consistent!");
      t.nextLap();
    }
    std::cout << "CPU OMP: " << t.lapAvg() << "+-" << t.lapStd() << " s"
              << std::endl;
    std::cout << "CPU OMP: " << (n / 1000.0 / 1000.0) / t.lapAvg()
              << " millions/s" << std::endl;
  }

  {
    // TODO: implement on OpenCL
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);
    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    gpu::gpu_mem_32u gpu_as, r;
    gpu_as.resizeN(n);
    gpu_as.writeN(as.data(), n);

    r.resizeN(1);

    ocl::Kernel sum(sum_kernel, sum_kernel_length, "sum");
    sum.compile(false);

    unsigned int workGroupSize = 128;
    unsigned int global_work_size =
        (n + workGroupSize - 1) / workGroupSize * workGroupSize;

    {
      timer t;
      for (int iter = 0; iter < benchmarkingIters; ++iter) {
        /** Потратил час, чтобы додуматься сделать это, даже стыдно ))) */
        unsigned int ssum = 0;
        r.writeN(&ssum, 1);
        sum.exec(gpu::WorkSize(workGroupSize, global_work_size), gpu_as, n, r);

        unsigned int gpu_sum;
        r.readN(&gpu_sum, 1);
        EXPECT_THE_SAME(reference_sum, gpu_sum,
                        "CPU OpenMP result should be consistent!");

        t.nextLap();
      }
      std::cout << "Opencl: " << t.lapAvg() << "+-" << t.lapStd() << " s"
                << std::endl;
      std::cout << "Opencl: " << (n / 1000.0 / 1000.0) / t.lapAvg()
                << " millions/s" << std::endl;
    }
  }
}
