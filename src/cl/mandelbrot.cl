#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void mandelbrot(__global float* results,
                         unsigned int width,
                         unsigned int height,
                         float from_x,
                         float from_y,
                         float size_x,
                         float size_y,
                         unsigned int iters,
                         unsigned int smoothing,
                         float threshold,
                         float threshold2)
{
    // TODO если хочется избавиться от зернистости и дрожжания при интерактивном погружении - добавьте anti-aliasing:
    // грубо говоря при anti-aliasing уровня N вам нужно рассчитать не одно значение в центре пикселя, а N*N значений
    // в узлах регулярной решетки внутри пикселя, а затем посчитав среднее значение результатов - взять его за результат для всего пикселя
    // это увеличит число операций в N*N раз, поэтому при рассчетах гигаплопс антиальясинг должен быть выключен
    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);
    if (width < i || height < j)
    {
        return;
    }
    float x0 = from_x + (i + 0.5f) * size_x / width;
    float y0 = from_y + (j + 0.5f) * size_y / height;

    float x = x0;
    float y = y0;

    int iter = 0;
    for (; iter < iters; ++iter)
    {
        float xPrev = x;
        x = x * x - y * y + x0;
        y = 2.0f * xPrev * y + y0;
        if ((x * x + y * y) > threshold2)
        {
            break;
        }
    }
    float result = iter;
    if (smoothing && iter != iters)
    {
        result = result - log(log(sqrt(x * x + y * y)) / log(threshold)) / log(2.0f);
    }

    result = 1.0f * result / iters;
    results[j * width + i] = result;
}
