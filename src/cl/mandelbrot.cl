#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

__kernel void mandelbrot(__global float *results,
                         const unsigned int width, const unsigned int height,
                         const float fromX, const float fromY,
                         const float sizeX, const float sizeY,
                         const int iters, const int smoothing) {
    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);
    if (i >= width || j >= height) {
        return;
    }
    const float threshold = 256.0f;
    const float threshold2 = threshold * threshold;
    float x0 = fromX + (i + 0.5f) * sizeX / width;
    float y0 = fromY + (j + 0.5f) * sizeY / height;
    float x = x0;
    float y = y0;
    int iteration = 0;
    for (; iteration < iters; ++iteration) {
        float tmpx = x;
        x = x * x - y * y + x0;
        y = 2.0f * tmpx * y + y0;
        if ((x * x + y * y) > threshold2) {
            break;
        }
    }
    float result = iteration;
    if (smoothing && iteration < iters) {
        result = result - log(log(sqrt(x * x + y * y)) / log(threshold)) / log(2.0f);
    }
    result = 1.0f * result / iters;
    results[j * width + i] = result;
}
