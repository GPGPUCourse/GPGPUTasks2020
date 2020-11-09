#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <iostream>
#include <iomanip>
#include "assert.h"

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/merge_cl.h"

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

using namespace std;

void printM(vector<float> as,
            vector<float> bs,
            vector<int> path) {

    cout << setw(4) << " ";
    for(int b = 0; b < bs.size(); b++) {
        cout << setw(4) << bs[b];
    }
    cout << endl;

    for(int a = 0; a < as.size(); a++) {
        for(int b = 0; b < bs.size(); b++) {

            if(b == 0) {
                cout << setw(4) << as[a];
            }
            int idx = a * bs.size() + b;

            cout << setw(4) << path[a * bs.size() + b];
        }
        cout << endl;
    }
}

int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    unsigned int n = 32 * 1024 * 1024;
    std::vector<float> as(n, 0);
    std::vector<float> bs(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.nextf();
        bs[i] = r.nextf();
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::sort(as.begin(), as.end());
    std::sort(bs.begin(), bs.end());

    std::vector<float> cpu_sorted;
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            cpu_sorted.resize(as.size() + bs.size());
            merge(as.begin(), as.end(), bs.begin(), bs.end(), cpu_sorted.begin());
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;
    }

    std::vector<float> merged;
    std::vector<int> path;

    merged.resize(n * 2);

    gpu::gpu_mem_32f as_gpu;
    as_gpu.resizeN(n);

    gpu::gpu_mem_32f bs_gpu;
    bs_gpu.resizeN(n);

    gpu::gpu_mem_32f merged_gpu;
    merged_gpu.resizeN(merged.size());
    {
        timer t;
        ocl::Kernel merge(merge_kernel, merge_kernel_length, "mergePath");
        merge.compile();

        ocl::Kernel mergeBlock(merge_kernel, merge_kernel_length, "mergeBlock");
        mergeBlock.compile();

        as_gpu.writeN(as.data(), as.size());
        bs_gpu.writeN(bs.data(), bs.size());

        unsigned int workGroupSize = 256;
        int NDRange = n / workGroupSize;
        unsigned int global_work_size = (NDRange + workGroupSize - 1) / workGroupSize * workGroupSize;

        path.resize(((n / workGroupSize) + 1) * 2);
        path.assign(path.size(), 0);

        gpu::gpu_mem_32i path_gpu;
        path_gpu.resizeN(path.size());

        path_gpu.writeN(path.data(), path.size());
        merged_gpu.writeN(merged.data(), merged.size());

        for (int iter = 0; iter < benchmarkingIters; ++iter) {

            merged_gpu.writeN(merged.data(), merged.size());
            t.restart(); // Запускаем секундомер после прогрузки данных чтобы замерять время работы кернела, а не трансфер данных
            merge.exec(gpu::WorkSize(workGroupSize, global_work_size),
                       as_gpu,
                       bs_gpu,
                       path_gpu,
                       as.size(),
                       (2*n) / NDRange);

            mergeBlock.exec(gpu::WorkSize(workGroupSize, NDRange + 1),
                            as_gpu,
                            bs_gpu,
                            merged_gpu,
                            path_gpu,
                            as.size());
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;
        merged_gpu.readN(merged.data(), merged.size());
    }
    // Проверяем корректность результатов
    for (int i = 0; i < merged.size(); ++i) {
        EXPECT_THE_SAME(merged[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}
