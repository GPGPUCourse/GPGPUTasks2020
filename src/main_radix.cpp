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

    {
        unsigned int WG_SIZE=256;
		unsigned int WG_COUNT=n/WG_SIZE;
		gpu::gpu_mem_32u tmp[2];
		tmp[0].resizeN(n);
		tmp[1].resizeN(n);
		gpu::gpu_mem_32u prefix_sums;
		prefix_sums.resizeN(WG_COUNT);

		ocl::Kernel radix(radix_kernel, radix_kernel_length, "radix");
		ocl::Kernel prefix_local(radix_kernel, radix_kernel_length, "prefix_local");
		ocl::Kernel prefix_global(radix_kernel, radix_kernel_length, "prefix_global");
		ocl::Kernel prefix_global_256(radix_kernel, radix_kernel_length, "prefix_global_256");
        radix.compile();
        prefix_local.compile();
        prefix_global.compile();
		prefix_global_256.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            tmp[0].writeN(as.data(), n);

            t.restart(); // Запускаем секундомер после прогрузки данных чтобы замерять время работы кернела, а не трансфер данных
            for(int bit_pos=0;bit_pos<32;bit_pos++)
			{
                prefix_local.exec(gpu::WorkSize(WG_SIZE, n), tmp[bit_pos%2], prefix_sums, n, bit_pos);
                prefix_global_256.exec(gpu::WorkSize(WG_SIZE, WG_COUNT), prefix_sums, WG_COUNT);
                for(int step=WG_SIZE;step<WG_COUNT;step*=2)
                    prefix_global.exec(gpu::WorkSize(WG_SIZE, WG_COUNT), prefix_sums, WG_COUNT, step);
                radix.exec(gpu::WorkSize(WG_SIZE, n), tmp[bit_pos%2], tmp[1-bit_pos%2], prefix_sums, n, bit_pos);
            }
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;

        tmp[0].readN(as.data(), n);
    }

    // Проверяем корректность результатов
	//for (int i = 0; i < n; ++i)
	//	std::cout<<as[i]<<','<<cpu_sorted[i]<<'\n';
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

	/*
	//optimize

	{
        unsigned int WG_SIZE=256;
		unsigned int WG_COUNT=n/WG_SIZE/2;
		gpu::gpu_mem_32u tmp[2];
		tmp[0].resizeN(n);
		tmp[1].resizeN(n);
		gpu::gpu_mem_32u prefix_sums;
		prefix_sums.resizeN(WG_COUNT);

		ocl::Kernel radix_2(radix_kernel, radix_kernel_length, "radix_2");
		ocl::Kernel prefix_local_2(radix_kernel, radix_kernel_length, "prefix_local_2");
		ocl::Kernel prefix_global_2(radix_kernel, radix_kernel_length, "prefix_global_2");
		ocl::Kernel prefix_global_256_2(radix_kernel, radix_kernel_length, "prefix_global_256_2");
        radix_2.compile();
        prefix_local_2.compile();
        prefix_global_2.compile();
		prefix_global_256_2.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            tmp[0].writeN(as.data(), n);

            t.restart(); // Запускаем секундомер после прогрузки данных чтобы замерять время работы кернела, а не трансфер данных
            for(unsigned int bit_pos=0;bit_pos<32;bit_pos++)
			{
                prefix_local_2.exec(gpu::WorkSize(WG_SIZE, n/2), tmp[bit_pos%2], prefix_sums, n, bit_pos);
                prefix_global_256_2.exec(gpu::WorkSize(WG_SIZE, WG_COUNT), prefix_sums, WG_COUNT);
                for(unsigned int step=WG_SIZE;step<WG_COUNT;step*=2)
                    prefix_global_2.exec(gpu::WorkSize(WG_SIZE, WG_COUNT), prefix_sums, WG_COUNT, step);
                radix_2.exec(gpu::WorkSize(WG_SIZE, n/2), tmp[bit_pos%2], tmp[1-bit_pos%2], prefix_sums, n, bit_pos);
            }
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;

        tmp[0].readN(as.data(), n);
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }
	*/

    return 0;
}
