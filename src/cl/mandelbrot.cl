#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

float calc_mandelbrot(const float x0, const float y0, const unsigned int iters) {
    const float threshold = 256.0f;
    const float threshold2 = threshold * threshold;
    
    float x = x0;
    float y = y0;

    int iter = 0;
    for (; iter < iters; ++iter) {
        float xPrev = x;
        x = x * x - y * y + x0;
        y = 2.0f * xPrev * y + y0;

        if ((x * x + y * y) > threshold2)
            break;
    }
    
    return 1.0f * iter / iters;
}

float smoothed_calc_mandelbrot(const float4 x0, const float4 y0, const unsigned int iters) {
    const float threshold = 256.0f;
    const float threshold2 = threshold * threshold;
    
    float4 x = x0;
    float4 y = y0;

    int4 iterCounts = { 0, 0, 0, 0 };
    int4 mask = { 1, 1, 1, 1 };

    for (int iter = 0; iter < iters; ++iter) {
        float4 xPrev = x;
        x = x * x - y * y + x0;
        y = 2.0f * xPrev * y + y0;

        mask &= ((x * x + y * y) <= threshold2);
        iterCounts += mask;

        if (all(mask))
            break;
    }
    
    return 0.25f * (iterCounts[0] + iterCounts[1] + iterCounts[2] + iterCounts[3]) / iters;
}

__kernel void mandelbrot(__global float* results, 
                   const unsigned int width, const unsigned int height,
                   const float fromX, const float fromY,
                   const float sizeX, const float sizeY,
                   const unsigned int iters, const int smoothing
) {
    
    const int i = get_global_id(0);
    const int j = get_global_id(1);

    if (i >= width || j >= height)
        return;

    const float stepX = sizeX * 1.0f / width;
    const float stepY = sizeY * 1.0f / height;

    const float x0 = fromX + (i + 0.5f) * stepX;
    const float y0 = fromY + (j + 0.5f) * stepY;

    if (smoothing) {
        // results[j * width + i] = (
        //     calc_mandelbrot(x0 - stepX / 2, y0 - stepX / 2, iters) + 
        //     calc_mandelbrot(x0 - stepX / 2, y0 + stepY / 2, iters) + 
        //     calc_mandelbrot(x0 + stepX / 2, y0 - stepX / 2, iters) + 
        //     calc_mandelbrot(x0 + stepX / 2, y0 + stepY / 2, iters)
        // ) / 4;
        const float4 smoothedX0 = { x0 - stepX / 2, x0 - stepX / 2, x0 + stepX / 2, x0 + stepX / 2 };
        const float4 smoothedY0 = { y0 - stepY / 2, y0 + stepY / 2, y0 - stepY / 2, y0 + stepY / 2 };
        results[j * width + i] = smoothed_calc_mandelbrot(smoothedX0, smoothedY0, iters); // not much difference compared to 4 direct computations 
                                                                                          // also tried using computed results, but it introduced artifacts
    } else {
        results[j * width + i] = calc_mandelbrot(x0, y0, iters);
    }
}
