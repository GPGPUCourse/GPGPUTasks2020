#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/sum_cl.h"

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


int main(int argc, char **argv) {
    int benchmarkingIters = 10;

    unsigned int reference_sum = 0;
    unsigned int n = 100 * 1000 * 1000;
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
        std::cout << "CPU:     " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
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
        std::cout << "CPU OMP: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        // chooseGPUDevice:
        // - Если не доступо ни одного устройства - кинет ошибку
        // - Если доступно ровно одно устройство - вернет это устройство
        // - Если доступно N>1 устройства:
        //   - Если аргументов запуска нет или переданное число не находится в диапазоне от 0 до N-1 - кинет ошибку
        //   - Если аргумент запуска есть и он от 0 до N-1 - вернет устройство под указанным номером
        gpu::Device device = gpu::chooseGPUDevice(argc, argv);

        // Этот контекст после активации будет прозрачно использоваться при всех вызовах в libgpu библиотеке
        // это достигается использованием thread-local переменных, т.е. на самом деле контекст будет активирован для текущего потока исполнения
        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();
        ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum");
        kernel.compile(false);
        gpu::gpu_mem_32u gpu_as, gpu_res;
        gpu_as.resizeN(n);
        gpu_res.resizeN(n);
        const unsigned int workGroupSize = 256;

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            gpu_as.writeN(as.data(), n);
            unsigned int curn = n;
            while (curn >= workGroupSize) {
                kernel.exec(gpu::WorkSize(workGroupSize, (curn + workGroupSize - 1) / workGroupSize * workGroupSize),
                            gpu_as, gpu_res, curn);
                gpu_as.swap(gpu_res);
                curn = (curn + workGroupSize - 1) / workGroupSize;
            }
            std::vector<unsigned int> results(curn);
            gpu_as.readN(results.data(), curn);
            unsigned int s = 0;
            for (unsigned int x : results) {
                s += x;
            }
            EXPECT_THE_SAME(reference_sum, s, "GPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }
}
