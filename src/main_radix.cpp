#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/radix_cl.h"

#include <numeric>
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

// Обозначения частично взяты из https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda
#define WORK_GROUP_SIZE 128
#define DATA_PER_WORKITEM 64 // количество элементов, которое обрабатывает один work_item
#define L 4  // количество одновременно сортируемых бит
#define PREFIXSUM_WG_NUM 16 // Количество рабочих групп для посчета префиксных сумм

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
            std::sort(cpu_sorted.begin(), cpu_sorted.end());
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;
    }

    gpu::gpu_mem_32u as_gpu;
    as_gpu.resizeN(n);

    // Constants
    // Рабочее пространство для фазы подсчета
    const unsigned int count_work_size = (n + WORK_GROUP_SIZE * DATA_PER_WORKITEM - 1) / (WORK_GROUP_SIZE * DATA_PER_WORKITEM)  * WORK_GROUP_SIZE;

    // Получившиеся количество рабочих групп в NDRange
    const unsigned int number_of_wg = count_work_size / WORK_GROUP_SIZE;

    // Размер масства подсчета для одной рабочей группы
    const unsigned int workgroup_count_size = (1 << L) * WORK_GROUP_SIZE;

    // Размер массива для подсчета по всем рабочим группам
    const unsigned int count_array_size = workgroup_count_size * number_of_wg;

    // Массив в видеопамяти для подсчета
    gpu::gpu_mem_32u count_array;
    count_array.resizeN(count_array_size);

    // Массив в видеопамяти для сохранения глобальных оффсетов
    gpu::gpu_mem_32u prefix_offset_array;
    prefix_offset_array.resizeN((1 << L));

    // Массив для хранения промежуточных результатов
    gpu::gpu_mem_32u swap_array;
    swap_array.resizeN(n);

    {
        ocl::Kernel radix_counter(radix_kernel, radix_kernel_length, "radix_counter");
        radix_counter.compile();

        ocl::Kernel radix_prefix_sum(radix_kernel, radix_kernel_length, "radix_prefix_sum");
        radix_prefix_sum.compile();

        ocl::Kernel radix_reorder(radix_kernel, radix_kernel_length, "radix_reorder");
        radix_reorder.compile();

        timer t;
        for (unsigned int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);
            t.restart(); // Запускаем секундомер после прогрузки данных чтобы замерять время работы кернела, а не трансфер данных

            for (unsigned int pass_number = 0; pass_number < (8 * sizeof(unsigned int)) / L; ++pass_number) {
                // Подсчет частот
                radix_counter.exec(gpu::WorkSize(WORK_GROUP_SIZE, count_work_size),
                                   as_gpu, count_array, n, pass_number);

                // Нахождения префиксных сумм
                radix_prefix_sum.exec(gpu::WorkSize(WORK_GROUP_SIZE, PREFIXSUM_WG_NUM * WORK_GROUP_SIZE),
                                      count_array, prefix_offset_array, count_array_size);

                // Перезаполнения массива
                radix_reorder.exec(gpu::WorkSize(WORK_GROUP_SIZE, count_work_size),
                                   as_gpu, swap_array, count_array, prefix_offset_array, n, pass_number);

                swap_array.swap(as_gpu);
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
