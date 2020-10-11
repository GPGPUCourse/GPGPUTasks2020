#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void mandelbrot(__global float *resultImg,const unsigned int width, const unsigned int height,
                        float fromX, float fromY,
                        float sizeX, float sizeY,
                        unsigned int cntIters, unsigned int smoothing){
    // TODO если хочется избавиться от зернистости и дрожания при интерактивном погружении, добавьте anti-aliasing:
    // грубо говоря, при anti-aliasing уровня N вам нужно рассчитать не одно значение в центре пикселя, а N*N значений
    // в узлах регулярной решетки внутри пикселя, а затем посчитав среднее значение результатов - взять его за результат для всего пикселя
    // это увеличит число операций в N*N раз, поэтому при рассчетах гигаплопс антиальясинг должен быть выключен



    const unsigned int idx = get_global_id(0);
    const unsigned int idy = get_global_id(1);

    if(idx >= width || idy >= height){
        return;
    }


    const float thr = 256.0f;
    const float logThr = 8;
    const float thrS = 256.0f * 256.0f;

    float x0 = fromX + (idx + 0.5f) * sizeX / width;

    float y0 = fromY + (idy + 0.5f) * sizeY / height;

    float x = x0;
    float y = y0;

    int i = 0;

    for(;i < cntIters;++i){
        float predX = x;

        x = x * x - y * y + x0;
        y = 2.0f * predX * y + y0;

        if((x * x + y * y) > thrS){
            break;
        }
    }

    float res = i;

    if(smoothing && (i != cntIters)){
        res -=  (log(log(sqrt(x * x + y * y)) / logThr));
    }

    res = (res * 1.0f) / cntIters;

    resultImg[idy * width + idx] = res;

}
