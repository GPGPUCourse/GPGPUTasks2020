#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#ifndef __OPENCL_C_VERSION__
  #include <cmath>
#endif

#line 10

#define GROUP_SIZE = 16;
__kernel void mandelbrot(__global float* results,
                                  unsigned int width, unsigned int height,
                                  float fromX, float fromY,
                                  float sizeX, float sizeY,
                                  unsigned int iters, unsigned int smoothing)
{
    // TODO если хочется избавиться от зернистости и дрожания при интерактивном погружении, добавьте anti-aliasing:
    // грубо говоря, при anti-aliasing уровня N вам нужно рассчитать не одно значение в центре пикселя, а N*N значений
    // в узлах регулярной решетки внутри пикселя, а затем посчитав среднее значение результатов - взять его за результат для всего пикселя
    // это увеличит число операций в N*N раз, поэтому при рассчетах гигаплопс антиальясинг должен быть выключен

    const float threshold = 256.0f;
    const float threshold2 = threshold * threshold;
    
    float x0 = fromX + (get_global_id(0) + 0.5f) * sizeX / width;
    float y0 = fromY + (get_global_id(1) + 0.5f) * sizeY / height;

    float x = x0, y = y0;

    unsigned int iter = 0;
    while (iter < iters) {
        float xPrev = x;
        x = x * x - y * y + x0;
        y = 2 * xPrev * y + y0;
        if (x * x + y * y > threshold2)
            break;
        iter++;
    }

    float result = iter;
    if (smoothing && iter != iters) {
        // result = result - logf(logf(sqrtf(x * x + y * y)) / logf(threshold)) / logf(2.0f);
    }

    result = 1.0f * result / iters;

    results[get_global_id(1) * width + get_global_id(0)] = result;
}
