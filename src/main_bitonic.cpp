#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include <libimages/images.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/bitonic_cl.h"

#include <vector>
#include <iostream>
#include <stdexcept>

#include <cassert>
#include <iomanip>

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << std::setprecision(10) << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

typedef unsigned int uint;

// check of as[start:end) is bitonic with peak at center
bool check_bitonic(const float* as, uint start, uint end) {
  bool bitonic = true;
  for (int i = start + 1; i < start + (end-start) / 2; ++i) {
    bitonic = bitonic && as[i - 1] <= as[i];
  }
  for (int i = start + (end-start) / 2 + 1; i < end; ++i) {
    bitonic = bitonic && as[i - 1] >= as[i];
  }
  return bitonic;
}

bool isPowerOf2(uint n) {
    if (n == 1) {
        return true;
    } else if (n & 1) {
        return false;
    } else {
        return isPowerOf2(n>>1);
    }
}

void red_sort(float* data, uint size, uint red_start, uint red_size) {
    assert(size > 1);
    assert(red_size > 1);
    assert(isPowerOf2(size));
    assert(isPowerOf2(red_size));
    assert(red_start + red_size <= size);
    uint i = red_start;
    uint j = i + red_size/2;
    for (; j < red_start + red_size; ++i, ++j) {
        float a_i = data[i];
        float a_j = data[j];
        if (a_i > a_j) {
            data[i] = a_j;
            data[j] = a_i;
        }
    }
}

void red_sort_reversed(float* data, uint size, uint red_start, uint red_size) {
    assert(size > 1);
    assert(red_size > 1);
    assert(isPowerOf2(size));
    assert(isPowerOf2(red_size));
    assert(red_start + red_size <= size);
    uint i = red_start;
    uint j = i + red_size/2;
    for (; j < red_start + red_size; ++i, ++j) {
        float a_i = data[i];
        float a_j = data[j];
        if (a_i < a_j) {
            data[i] = a_j;
            data[j] = a_i;
        }
    }
}

typedef void (*RedSortType)(float*, uint, uint, uint);

void do_block(float* data, uint size, uint block_start, uint block_size, RedSortType red_func) {
    assert(block_start % block_size == 0);
    if (red_func == red_sort_reversed) {
        assert((block_start - block_size) % (2*block_size) == 0);
    } else {
        assert(block_start % (2*block_size) == 0);
    }
    assert(isPowerOf2(block_size));
    assert(isPowerOf2(size));
    for (uint red_size = block_size; red_size > 1; red_size >>= 1) {
        const uint last_red_start = block_start + block_size - red_size;
        for (uint red_start = block_start; red_start <= last_red_start; ++red_start) {
            red_func(data, size, red_start, red_size);
        }
    }
}

// bitonic on CPU
void bitonic_sort(float* data, uint size) {
    assert(isPowerOf2(size));
    for (uint block_size = 2; block_size <= size; block_size <<= 1) {
        const uint twice_block_size = 2*block_size;
        // for (uint i = 0; i < size; ++i) std::cout << ", " << data[i]; std::cout << "\n";
        for (uint block_start = 0; block_start < size; block_start += twice_block_size) {
            do_block(data, size, block_start, block_size, red_sort);
        }
        for (uint block_start = block_size; block_start < size; block_start += twice_block_size) {
            do_block(data, size, block_start, block_size, red_sort_reversed);
        }
    }
}

// render graph of sequence
void renderInWindow(const std::vector<float>& as);


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
        ocl::Kernel bitonic_begin512(bitonic_kernel, bitonic_kernel_length, "bitonic_begin512");
        bitonic_begin512.compile();
        ocl::Kernel bitonic_step(bitonic_kernel, bitonic_kernel_length, "bitonic_step");
        bitonic_step.compile();
        ocl::Kernel bitonic_finisher(bitonic_kernel, bitonic_kernel_length, "bitonic_finisher");
        bitonic_finisher.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart(); // Запускаем секундомер после прогрузки данных чтобы замерять время работы кернела, а не трансфер данных
            bitonic_begin512.exec(gpu::WorkSize(128, n/4), as_gpu);

            uint block_size_log = 9;
            uint block_size = 512;
            for (; block_size <= n/2; block_size <<= 1, ++block_size_log) {
                for (uint swap_dist_logPLUS = block_size_log; swap_dist_logPLUS >= 1; --swap_dist_logPLUS) {
                    bitonic_step.exec(gpu::WorkSize(128, n/4), as_gpu, block_size, swap_dist_logPLUS-1);
                }
            }
            // block_size = n, block_size_log = log2(n)
            for (uint swap_dist_logPLUS = block_size_log; swap_dist_logPLUS >= 1; --swap_dist_logPLUS) {
                bitonic_finisher.exec(gpu::WorkSize(128, n/2), as_gpu, block_size, swap_dist_logPLUS-1);
            }

            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;

        as_gpu.readN(as.data(), n);
    }


    // Проверяем корректность результатов
    // renderInWindow(as);
    bool sorted = true;
    for (uint i = 1; i < n; ++i) {
      sorted = sorted && as[i-1] <= as[i];
    }
    std::cout << "sorted: " << sorted << "\n";

    //std::sort(as.begin(), as.end()); // для проверки на случай повреждения массива
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }
    return 0;
}

float vec_min(const std::vector<float>& as) {
    float m = as[0];
    for (float v : as) m = m > v ? v : m;
    return m;
}

float vec_max(const std::vector<float>& as) {
    float M = as[0];
    for (float v : as) M = M < v ? v : M;
    return M;
}

void renderInWindow(const std::vector<float>& as)
{
    images::ImageWindow window("Mandelbrot");

    unsigned int width = 1024;
    unsigned int height = 1024;

    float sizeX = 2.0f;
    float sizeY = sizeX * height / width;

    images::Image<unsigned char> image(width, height, 3);

    // image.fill(0) и image.fill('\0') не работают!
    // хотя image.fill('0') как и image.fill(1) почему-то работают:
    //     Ошибка(активно) E0308 существует более одного экземпляра перегруженная функция
    //     "images::Image<T>::fill [с T=unsigned char]", соответствующего списку аргументов

    // image.fill(0);
    for (size_t y = 0; y < height; y++) {
      for (size_t x = 0; x < width; x++) {
        for (size_t c = 0; c < 3; c++) {
          image(y, x, c) = 0;
        }
      }
    }

    float a_min = vec_min(as);
    float a_max = vec_max(as);

    auto map = [](float n, float m, float M) -> int {
        return (n-m)/(M-m)*1024.0f;
    };

    for (int i = 0; i < 1024; ++i) {
        int h = map(as[i*as.size()/1024], a_min, a_max);
        for (int j = 0; j < h; ++j) {
            image(1023 - j, i, 0) = (unsigned char) (0.0 * 255);
            image(1023 - j, i, 1) = (unsigned char) (1.0 * 255);
            image(1023 - j, i, 2) = (unsigned char) (0.0 * 255);
        }
    }

    do {
        window.display(image);
        window.wait(30);
    } while (!window.isClosed());
}
