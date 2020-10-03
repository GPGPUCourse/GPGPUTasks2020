#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void mandelbrot(__global float *image, unsigned int rows,
                         unsigned int cols, float fromX, float fromY,
                         float sizeX, float sizeY, unsigned int iters,
                         int smoothing) {

  // TODO если хочется избавиться от зернистости и дрожжания при интерактивном
  // погружении - добавьте anti-aliasing: грубо говоря при anti-aliasing уровня
  // N вам нужно рассчитать не одно значение в центре пикселя, а N*N значений в
  // узлах регулярной решетки внутри пикселя, а затем посчитав среднее значение
  // результатов - взять его за результат для всего пикселя это увеличит число
  // операций в N*N раз, поэтому при рассчетах гигаплопс антиальясинг должен
  // быть выключен
  int i = get_global_id(0);
  int j = get_global_id(1);
  float thresh = 256 * 256;

  if (j * (int)cols + i >= rows * cols)
    return;

  float x0 = fromX + (i + 0.5f) * sizeX / cols;
  float y0 = fromY + (j + 0.5f) * sizeY / rows;

  float x = x0;
  float y = y0;

  int iter = 0;
  for (; iter < iters; ++iter) {
    float xPrev = x;
    x = x * x - y * y + x0;
    y = 2.0f * xPrev * y + y0;
    if ((x * x + y * y) > thresh) {
      break;
    }
  }

  float result = iter;
  if (smoothing == 1 && iter != iters) {
    result = result - log(log(sqrt(x * x + y * y)) / log(thresh)) / log(2.0f);
  }

  result = 1.0f * result / iters;
  image[j * (int)cols + i] = result;
}
