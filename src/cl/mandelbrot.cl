#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void mandelbrot(__global float *results,
        unsigned int width,
        unsigned int height,
        loat fromX,
        float fromY,
        float sizeX,
        float sizeY,
        unsigned int iters,
        unsigned int smoothing)
{
    const float threshold = 256.0f;
    const float threshold2 = threshold * threshold;

    if (width <= get_global_id(0) || height <= get_global_id(1)) {
        return;
    }

    float x0 = fromX + (get_global_id(0) + 0.5f) * sizeX / width;
    float y0 = fromY + (get_global_id(1) + 0.5f) * sizeY / height;

    float x = x0;
    float y = y0;

    int iter = 0;
    for (; iter < iters; ++iter) {
        float xPrev = x;
        x = x * x - y * y + x0;
        y = 2.0f * xPrev * y + y0;
        if ((x * x + y * y) > threshold2) {
            break;
        }
    }
    float result = iter;

    if (smoothing && iter != iters) {
        result = result - log(log(sqrt(x * x + y * y)) / log(threshold)) / log(2.0f);
    }

    result = 1.0f * result / iters;
    results[get_global_id(1) * width + get_global_id(0)] = result;
}
