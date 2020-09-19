#include <CL/cl.h>
#include <libclew/ocl_init.h>
#include <libutils/fast_random.h>
#include <libutils/timer.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

template <typename T> std::string to_string(T value) {
  std::ostringstream ss;
  ss << value;
  return ss.str();
}

void reportError(cl_int err, const std::string &filename, int line) {
  if (CL_SUCCESS == err)
    return;

  // Таблица с кодами ошибок:
  // libs/clew/CL/cl.h:103
  // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с
  // номером строки: cl.h:103) -> Enter
  std::string message = "OpenCL error code " + to_string(err) +
                        " encountered at " + filename + ":" + to_string(line);
  throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)

void pfn_notify(const char *errinfo, const void *private_info, size_t cb,
                void *user_data) {
  std::string message = "OpenCL Error \n" + std::string(errinfo);
  throw std::runtime_error(message);
}

// IDK if CI gcc from `The Mesozoic Era` supports std::optional, so i'll return
// pair
//
// i use g++-10 as daily compiler
std::pair<cl_device_id, bool>
getDevice(cl_device_type type, const std::vector<cl_platform_id> &platforms) {
  cl_device_id device;
  bool found = false;
  // Choose "random" (last) videocard
  for (auto platform : platforms) {
    cl_uint devicesCount = 0;

    OCL_SAFE_CALL(clGetDeviceIDs(platform, type, 0, NULL, &devicesCount));

    if (devicesCount == 0)
      continue;

    std::vector<cl_device_id> devices(devicesCount);
    OCL_SAFE_CALL(
        clGetDeviceIDs(platform, type, devicesCount, devices.data(), NULL));

    device = devices[0];
    found = true;
  }
  return {device, found};
}

int main() {
  // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку
  // clew)
  if (!ocl_init())
    throw std::runtime_error("Can't init OpenCL driver!");

  // TODO 1 По аналогии с предыдущим заданием узнайте какие есть устройства, и
  // выберите из них какое-нибудь (если в списке устройств есть хоть одна
  // видеокарта - выберите ее, если нету - выбирайте процессор)
  cl_uint platformsCount = 0;
  OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
  std::vector<cl_platform_id> platforms(platformsCount);
  OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));

  cl_device_id device;
  bool gpu = true;
  auto get_GPU_device = getDevice(CL_DEVICE_TYPE_GPU, platforms);
  if (get_GPU_device.second)
    device = get_GPU_device.first;
  else {
    gpu = false;
    auto get_CPU_device = getDevice(CL_DEVICE_TYPE_CPU, platforms);
    if (!get_CPU_device.second) {
      throw std::runtime_error("Can't find device");
    }
    device = get_CPU_device.first;
  }

  // TODO 2 Создайте контекст с выбранным устройством
  // См. документацию
  // https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL
  // Runtime -> Contexts -> clCreateContext Не забывайте проверять все
  // возвращаемые коды на успешность (обратите внимание что в данном случае
  // метод возвращает код по переданному аргументом errcode_ret указателю) И
  // хорошо бы сразу добавить в конце clReleaseContext (да, не очень RAII, но
  // это лишь пример)
  cl_int context_err;
  cl_context context =
      clCreateContext(NULL, 1, &device, &pfn_notify, NULL, &context_err);
  OCL_SAFE_CALL(context_err);

  // TODO 3 Создайте очередь выполняемых команд в рамках выбранного контекста и
  // устройства См. документацию
  // https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL
  // Runtime -> Runtime APIs -> Command Queues -> clCreateCommandQueue Убедитесь
  // что в соответствии с документацией вы создали in-order очередь задач И
  // хорошо бы сразу добавить в конце clReleaseQueue (не забывайте освобождать
  // ресурсы)

  cl_int queue_err;
  cl_command_queue queue_cl =
      clCreateCommandQueue(context, device, 0, &queue_err);
  OCL_SAFE_CALL(queue_err);

  unsigned int n = 100 * 1000 * 1000;
  // Создаем два массива псевдослучайных данных для сложения и массив для
  // будущего хранения результата
  std::vector<float> as(n, 0);
  std::vector<float> bs(n, 0);
  std::vector<float> cs(n, 0);
  FastRandom r(n);
  for (unsigned int i = 0; i < n; ++i) {
    as[i] = r.nextf();
    bs[i] = r.nextf();
  }
  std::cout << "Data generated for n=" << n << "!" << std::endl;

  // TODO 4 Создайте три буфера в памяти устройства (в случае видеокарты - в
  // видеопамяти - VRAM) - для двух суммируемых массивов as и bs (они read-only)
  // и для массива с результатом cs (он write-only) См. Buffer Objects ->
  // clCreateBuffer Размер в байтах соответственно можно вычислить через
  // sizeof(float)=4 и тот факт что чисел в каждом массиве - n штук Данные в as
  // и bs можно прогрузить этим же методом скопировав данные из
  // host_ptr=as.data() (и не забыв про битовый флаг на это указывающий) или же
  // через метод Buffer Objects -> clEnqueueWriteBuffer И хорошо бы сразу
  // добавить в конце clReleaseMemObject (аналогично все дальнейшие ресурсы
  // вроде OpenCL под-программы, кернела и т.п. тоже нужно освобождать)

  cl_int mem_err;
  cl_mem a_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               n * sizeof(float), as.data(), &mem_err);
  OCL_SAFE_CALL(mem_err);

  cl_mem b_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               n * sizeof(float), bs.data(), &mem_err);
  OCL_SAFE_CALL(mem_err);

  cl_mem c_cl = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n * sizeof(float),
                               NULL, &mem_err);
  OCL_SAFE_CALL(mem_err);

  // TODO 6 Выполните TODO 5 (реализуйте кернел в src/cl/aplusb.cl)
  // затем убедитесь что выходит загрузить его с диска (убедитесь что Working
  // directory выставлена правильно - см. описание задания) напечатав исходники
  // в консоль (if проверяет что удалось считать хоть что-то)
  std::string kernel_sources;
  {
    std::ifstream file("src/cl/aplusb.cl");
    kernel_sources = std::string(std::istreambuf_iterator<char>(file),
                                 std::istreambuf_iterator<char>());
    if (kernel_sources.size() == 0) {
      throw std::runtime_error("Empty source file! May be you forgot to "
                               "configure working directory properly?");
    }
    std::cout << "\t KERNEL SOURCES:\n\n";
    std::cout << kernel_sources << std::endl << std::endl;
  }

  // TODO 7 Создайте OpenCL-подпрограмму с исходниками кернела
  // см. Runtime APIs -> Program Objects -> clCreateProgramWithSource
  // у string есть метод c_str(), но обратите внимание что передать вам нужно
  // указатель на указатель

  cl_int program_err;
  const char *dummy = {kernel_sources.c_str()};
  const char *strings[1] = {dummy};
  size_t lengths[1] = {kernel_sources.size()};
  cl_program program =
      clCreateProgramWithSource(context, 1, strings, lengths, &program_err);
  OCL_SAFE_CALL(program_err);

  // TODO 8 Теперь скомпилируйте программу и напечатайте в консоль лог
  // компиляции см. clBuildProgram

  cl_int build_err = clBuildProgram(program, 1, &device, "", NULL, NULL);

  // А так же напечатайте лог компиляции (он будет очень полезен, если в кернеле
  // есть синтаксические ошибки - т.е. когда clBuildProgram вернет
  // CL_BUILD_PROGRAM_FAILURE) Обратите внимание что при компиляции на
  // процессоре через Intel OpenCL драйвер - в логе указывается какой ширины
  // векторизацию получилось выполнить для кернела см. clGetProgramBuildInfo
  size_t log_size = 0;
  OCL_SAFE_CALL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0,
                                      NULL, &log_size));

  std::vector<char> log(log_size, 0);

  OCL_SAFE_CALL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                                      log_size, log.data(), NULL));
  // У меня если нет ошибок, то лог пустой ..
  if (log_size > 1) {
    std::cout << "Log:" << std::endl;
    std::cout << log.data() << std::endl;
  }
  OCL_SAFE_CALL(build_err);

  // TODO 9 Создайте OpenCL-kernel в созданной подпрограмме (в одной
  // подпрограмме может быть несколько кернелов, но в данном случае кернел один)
  // см. подходящую функцию в Runtime APIs -> Program Objects -> Kernel Objects
  cl_int kernel_err;

  cl_kernel kernel = clCreateKernel(program, "aplusb", &kernel_err);
  OCL_SAFE_CALL(kernel_err);

  // TODO 10 Выставите все аргументы в кернеле через clSetKernelArg (as_gpu,
  // bs_gpu, cs_gpu и число значений, убедитесь что тип количества элементов
  // такой же в кернеле)
  {
    OCL_SAFE_CALL(clSetKernelArg(kernel, 0, sizeof(a_cl), &a_cl));
    OCL_SAFE_CALL(clSetKernelArg(kernel, 1, sizeof(b_cl), &b_cl));
    OCL_SAFE_CALL(clSetKernelArg(kernel, 2, sizeof(c_cl), &c_cl));
    OCL_SAFE_CALL(clSetKernelArg(kernel, 3, sizeof(n), &n));
  }

  // TODO 11 Выше увеличьте n с 1000*1000 до 100*1000*1000 (чтобы дальнейшие
  // замеры были ближе к реальности)

  // Честно сделал

  // TODO 12 Запустите выполнения кернела:
  // - С одномерной рабочей группой размера 128
  // - В одномерном рабочем пространстве размера roundedUpN, где roundedUpN -
  // наименьшее число кратное 128 и при этом не меньшее n
  // - см. clEnqueueNDRangeKernel
  // - Обратите внимание что чтобы дождаться окончания вычислений (чтобы знать
  // когда можно смотреть результаты в cs_gpu) нужно:
  //   - Сохранить событие "кернел запущен" (см. аргумент "cl_event *event")
  //   - Дождаться завершения полунного события - см. в документации подходящий
  //   метод среди Event Objects
  {
    size_t workGroupSize = 128;
    size_t global_work_size =
        (n + workGroupSize - 1) / workGroupSize * workGroupSize;
    timer t; // Это вспомогательный секундомер, он замеряет время своего
             // создания и позволяет усреднять время нескольких замеров
    for (unsigned int i = 0; i < 20; ++i) {
      cl_event event;
      OCL_SAFE_CALL(clEnqueueNDRangeKernel(queue_cl, kernel, 1, NULL,
                                           &global_work_size, &workGroupSize, 0,
                                           NULL, &event));
      OCL_SAFE_CALL(clWaitForEvents(1, &event));
      t.nextLap(); // При вызове nextLap секундомер запоминает текущий замер
                   // (текущий круг) и начинает замерять время следующего круга
    }
    // Среднее время круга (вычисления кернела) на самом деле считаются не по
    // всем замерам, а лишь с 20%-перцентайля по 80%-перцентайль (как и
    // стандартное отклониение) подробнее об этом - см. timer.lapsFiltered P.S.
    // чтобы в CLion быстро перейти к символу (функции/классу/много чему еще)
    // достаточно нажать Ctrl+Shift+Alt+N -> lapsFiltered -> Enter
    std::cout << "Kernel average time: " << t.lapAvg() << "+-" << t.lapStd()
              << " s" << std::endl;

    // TODO 13 Рассчитайте достигнутые гигафлопcы:
    // - Всего элементов в массивах по n штук
    // - Всего выполняется операций: операция a+b выполняется n раз
    // - Флопс - это число операций с плавающей точкой в секунду
    // - В гигафлопсе 10^9 флопсов
    // - Среднее время выполнения кернела равно t.lapAvg() секунд
    //
    // Если я верно всё понимаю, то число элементов в массиве у нас это
    // global_work_size, т.к. в самой последней ворк группе всеравно будет
    // затрачено времени как на вычисление workGroupSize сложений ?
    std::cout << "GFlops: " << (global_work_size / t.lapAvg() / 1e9)
              << std::endl;

    // TODO 14 Рассчитайте используемую пропускную способность обращений к
    // видеопамяти (в гигабайтах в секунду)
    // - Всего элементов в массивах по n штук
    // - Размер каждого элемента sizeof(float)=4 байта
    // - Обращений к видеопамяти т.о. 2*n*sizeof(float) байт на чтение и
    // 1*n*sizeof(float) байт на запись, т.е. итого 3*n*sizeof(float) байт
    // - В гигабайте 1024*1024*1024 байт
    // - Среднее время выполнения кернела равно t.lapAvg() секунд
    std::cout << "VRAM bandwidth: "
              << (3 * n * sizeof(float) / t.lapAvg() / 1024 / 1024 / 1024)
              << " GB/s" << std::endl;
  }

  // TODO 15 Скачайте результаты вычислений из видеопамяти (VRAM) в оперативную
  // память (RAM) - из cs_gpu в cs (и рассчитайте скорость трансфера данных в
  // гигабайтах в секунду)
  {
    timer t;
    for (unsigned int i = 0; i < 20; ++i) {
      cl_event event;
      OCL_SAFE_CALL(clEnqueueReadBuffer(queue_cl, c_cl, CL_TRUE, 0,
                                        n * sizeof(float), cs.data(), 0, NULL,
                                        &event));
      OCL_SAFE_CALL(clWaitForEvents(1, &event));
      t.nextLap();
    }
    std::cout << "Result data transfer time: " << t.lapAvg() << "+-"
              << t.lapStd() << " s" << std::endl;
    std::cout << "VRAM -> RAM bandwidth: "
              << (n * sizeof(float) / t.lapAvg() / 1024 / 1024 / 1024)
              << " GB/s" << std::endl;
  }

  // TODO 16 Сверьте результаты вычислений со сложением чисел на процессоре (и
  // убедитесь, что если в кернеле сделать намеренную ошибку, то эта проверка
  // поймает ошибку)
  //
  // Честно попробовал :)
  for (unsigned int i = 0; i < n; ++i) {
    if (cs[i] != as[i] + bs[i]) {
      throw std::runtime_error("CPU and GPU results differ!");
    }
  }

  OCL_SAFE_CALL(clReleaseCommandQueue(queue_cl));
  OCL_SAFE_CALL(clReleaseContext(context));
  OCL_SAFE_CALL(clReleaseMemObject(a_cl));
  OCL_SAFE_CALL(clReleaseMemObject(b_cl));
  OCL_SAFE_CALL(clReleaseMemObject(c_cl));
  OCL_SAFE_CALL(clReleaseProgram(program));
  OCL_SAFE_CALL(clReleaseKernel(kernel));
  return 0;
}
