#include <CL/cl.h>
#include <libclew/ocl_init.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>

#include <vector>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <cassert>


template <typename T>
std::string to_string(T value)
{
    std::ostringstream ss;
    ss << value;
    return ss.str();
}

void reportError(cl_int err, const std::string &filename, int line)
{
    if (CL_SUCCESS == err)
        return;

    // Таблица с кодами ошибок:
    // libs/clew/CL/cl.h:103
    // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
    std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
    throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)

class context final
{
private:
  cl_context Context = nullptr;
  cl_device_id SelectedDevice = nullptr;

  context( const context & ) = delete;
  context & operator=( const context & ) = delete;

  static void CL_CALLBACK ErrorCallback( const char *ErrInfo, const void *, size_t, void * )
  {
    std::cout << "Error callback param: " << ErrInfo << std::endl;
    throw std::runtime_error(ErrInfo);
  }

public:
  context( void )
  {
    bool IsSelected = false;
    cl_uint PlatformsCount = 0;
    std::vector<cl_platform_id> Platforms(PlatformsCount);

    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &PlatformsCount));
    Platforms.resize(PlatformsCount);
    OCL_SAFE_CALL(clGetPlatformIDs(PlatformsCount, Platforms.data(), nullptr));

    for (unsigned int PlatformIndex = 0; PlatformIndex < PlatformsCount; ++PlatformIndex)
    {
      cl_platform_id Platform = Platforms[PlatformIndex];;

      cl_uint DevicesCount = 0;
      std::vector<cl_device_id> Devices(DevicesCount);

      OCL_SAFE_CALL(clGetDeviceIDs(Platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &DevicesCount));
      Devices.resize(DevicesCount);
      OCL_SAFE_CALL(clGetDeviceIDs(Platform, CL_DEVICE_TYPE_ALL, DevicesCount, Devices.data(), nullptr));

      for (unsigned int DeviceIndex = 0; DeviceIndex < DevicesCount; ++DeviceIndex)
      {
        cl_device_id Device = Devices[DeviceIndex];

        if (SelectedDevice == nullptr)
          SelectedDevice = Device;

        size_t DeviceTypeSize = 0;
        cl_device_type DeviceType;

        OCL_SAFE_CALL(clGetDeviceInfo(Device, CL_DEVICE_TYPE, 0, nullptr, &DeviceTypeSize));
        OCL_SAFE_CALL(clGetDeviceInfo(Device, CL_DEVICE_TYPE, DeviceTypeSize, &DeviceType, nullptr));
        if (DeviceType & CL_DEVICE_TYPE_GPU)
        {
          IsSelected = true;
          SelectedDevice = Device;
          break;
        }
      }

      if (IsSelected)
        break;
    }

    cl_int Error = CL_SUCCESS;

    Context = clCreateContext(nullptr, 1, &SelectedDevice, ErrorCallback, nullptr, &Error);
    OCL_SAFE_CALL(Error);

    //size_t DeviceNameSize = 0;
    //std::vector<unsigned char> DeviceName(DeviceNameSize, 0);
    //
    //OCL_SAFE_CALL(clGetDeviceInfo(SelectedDevice, CL_DEVICE_NAME, 0, nullptr, &DeviceNameSize));
    //DeviceName.resize(DeviceNameSize);
    //OCL_SAFE_CALL(clGetDeviceInfo(SelectedDevice, CL_DEVICE_NAME, DeviceNameSize, DeviceName.data(), nullptr));
    //std::cout << "Selected device: " << DeviceName.data() << "\n";
  }

  cl_context GetContext( void ) const
  {
    return Context;
  }

  cl_device_id GetDevice( void ) const
  {
    return SelectedDevice;
  }

  ~context( void )
  {
    OCL_SAFE_CALL(clReleaseContext(Context));
  }
};

class queue
{
private:
  cl_command_queue Queue = nullptr;

  queue( const queue & ) = delete;
  queue & operator=( const queue & ) = delete;

public:
  queue( const context &Ctx )
  {
    cl_int Error = CL_SUCCESS;
    Queue = clCreateCommandQueue(Ctx.GetContext(), Ctx.GetDevice(), 0, &Error);
    OCL_SAFE_CALL(Error);
  }

  cl_command_queue GetQueue( void ) const
  {
    return Queue;
  }

  ~queue( void )
  {
    OCL_SAFE_CALL(clReleaseCommandQueue(Queue));
  }
};

class buffer
{
private:
  cl_mem Buffer = nullptr;

  buffer( const buffer & ) = delete;
  buffer & operator=( const buffer & ) = delete;

public:
  buffer( const context &Ctx, cl_mem_flags Flags, size_t Size, void *HostPtr = nullptr )
  {
    cl_int Error = CL_SUCCESS;
    Buffer = clCreateBuffer(Ctx.GetContext(), Flags, Size, HostPtr, &Error);
    OCL_SAFE_CALL(Error);
  }

  void CopyDataToBuffer( const queue &Queue, const void *Data, size_t Size, size_t Offset = 0,
                         const bool IsBlocking = true,
                         const std::vector<cl_event> &WaitEvents = std::vector<cl_event>(),
                         cl_event *Event = nullptr )
  {

    OCL_SAFE_CALL(clEnqueueWriteBuffer(Queue.GetQueue(), Buffer, IsBlocking, Offset,
                                       Size, Data, (cl_uint)WaitEvents.size(),
                                       WaitEvents.data(), Event));
  }

  void CopyDataFromBuffer( const queue &Queue, void *Data, size_t Size, size_t Offset = 0,
                           const bool IsBlocking = true,
                           const std::vector<cl_event> &WaitEvents = std::vector<cl_event>(),
                           cl_event *Event = nullptr )
  {
    OCL_SAFE_CALL(clEnqueueReadBuffer(Queue.GetQueue(), Buffer, IsBlocking, Offset,
                                      Size, Data, (cl_uint)WaitEvents.size(),
                                      WaitEvents.data(), Event));
  }

  const cl_mem * GetBufferPtr( void ) const
  {
    return &Buffer;
  }

  ~buffer( void )
  {
    OCL_SAFE_CALL(clReleaseMemObject(Buffer));
  }
};

class program
{
private:
  cl_kernel Kernel = nullptr;
  cl_program Program = nullptr;
  const context &Ctx;

  program( const program & ) = delete;
  program & operator=( const program & ) = delete;

public:
  program( const context &Ctx, const std::string &Src ) : Ctx(Ctx)
  {
    cl_int Error = CL_SUCCESS;
    const char *SrcData = Src.c_str();
    const size_t Size = Src.size();

    Program = clCreateProgramWithSource(Ctx.GetContext(), 1, &SrcData, &Size, &Error);
    OCL_SAFE_CALL(Error);

    cl_device_id Device = Ctx.GetDevice();

    OCL_SAFE_CALL(clBuildProgram(Program, 1, &Device, nullptr, nullptr, nullptr));
  }

  cl_int CreateKernel( const std::string &Name )
  {
    cl_int Error = CL_SUCCESS;
    
    Kernel = clCreateKernel(Program, Name.c_str(), &Error);

    return Error;
  }

  void GetBuildLog( std::vector<char> *Str )
  {
    size_t Size = 0;
    
    OCL_SAFE_CALL(clGetProgramBuildInfo(Program, Ctx.GetDevice(), CL_PROGRAM_BUILD_LOG,
                                        0, nullptr, &Size));
    Str->resize(Size);
    OCL_SAFE_CALL(clGetProgramBuildInfo(Program, Ctx.GetDevice(), CL_PROGRAM_BUILD_LOG,
                                        Size, Str->data(), nullptr));
  }

  void SetArg( cl_uint Index, size_t Size, const void *Value )
  {
    OCL_SAFE_CALL(clSetKernelArg(Kernel, Index, Size, Value));
  }

  void Run1D( const queue &Queue, const size_t GlobalWorkOffset,
              const size_t GlobalWorkSize, const size_t LocalWorkSize )
  {
    cl_event Event = nullptr;

    OCL_SAFE_CALL(clEnqueueNDRangeKernel(Queue.GetQueue(), Kernel, 1,
      &GlobalWorkOffset, &GlobalWorkSize, &LocalWorkSize, 0, nullptr, &Event));

    OCL_SAFE_CALL(clWaitForEvents(1, &Event));
  }

  ~program( void )
  {
    OCL_SAFE_CALL(clReleaseKernel(Kernel));
    OCL_SAFE_CALL(clReleaseProgram(Program));
  }
};

int main()
{
  try
  {
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку clew)
    if (!ocl_init())
      throw std::runtime_error("Can't init OpenCL driver!");

    // TODO 1 По аналогии с предыдущим заданием узнайте какие есть устройства, и выберите из них какое-нибудь
    // (если в списке устройств есть хоть одна видеокарта - выберите ее, если нету - выбирайте процессор)

    // TODO 2 Создайте контекст с выбранным устройством
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Contexts -> clCreateContext
    // Не забывайте проверять все возвращаемые коды на успешность (обратите внимание что в данном случае метод возвращает
    // код по переданному аргументом errcode_ret указателю)
    // И хорошо бы сразу добавить в конце clReleaseContext (да, не очень RAII, но это лишь пример)
    context Context;
    {
      // TODO 3 Создайте очередь выполняемых команд в рамках выбранного контекста и устройства
      // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Runtime APIs -> Command Queues -> clCreateCommandQueue
      // Убедитесь что в соответствии с документацией вы создали in-order очередь задач
      // И хорошо бы сразу добавить в конце clReleaseQueue (не забывайте освобождать ресурсы)
      queue Queue{Context};

      unsigned int n = 1000 * 1000 * 100;
      // Создаем два массива псевдослучайных данных для сложения и массив для будущего хранения результата
      std::vector<float> as(n, 0);
      std::vector<float> bs(n, 0);
      std::vector<float> cs(n, 0);
      FastRandom r(n);
      for (unsigned int i = 0; i < n; ++i)
      {
        as[i] = r.nextf();
        bs[i] = r.nextf();
      }
      std::cout << "Data generated for n=" << n << "!" << std::endl;

      // TODO 4 Создайте три буфера в памяти устройства (в случае видеокарты - в видеопамяти - VRAM) - для двух суммируемых массивов as и bs (они read-only) и для массива с результатом cs (он write-only)
      // См. Buffer Objects -> clCreateBuffer
      // Размер в байтах соответственно можно вычислить через sizeof(float)=4 и тот факт что чисел в каждом массиве - n штук
      // Данные в as и bs можно прогрузить этим же методом скопировав данные из host_ptr=as.data() (и не забыв про битовый флаг на это указывающий)
      // или же через метод Buffer Objects -> clEnqueueWriteBuffer
      // И хорошо бы сразу добавить в конце clReleaseMemObject (аналогично все дальнейшие ресурсы вроде OpenCL под-программы, кернела и т.п. тоже нужно освобождать)
      buffer BufferA{Context, CL_MEM_READ_ONLY, n * sizeof(float)};
      buffer BufferB{Context, CL_MEM_READ_ONLY, n * sizeof(float)};
      buffer BufferC{Context, CL_MEM_WRITE_ONLY, n * sizeof(float)};

      BufferA.CopyDataToBuffer(Queue, as.data(), as.size() * sizeof(float));
      BufferB.CopyDataToBuffer(Queue, bs.data(), bs.size() * sizeof(float));

      // TODO 6 Выполните TODO 5 (реализуйте кернел в src/cl/aplusb.cl)
      // затем убедитесь что выходит загрузить его с диска (убедитесь что Working directory выставлена правильно - см. описание задания)
      // напечатав исходники в консоль (if проверяет что удалось считать хоть что-то)
      std::string kernel_sources;
      {
        std::ifstream file("src/cl/aplusb.cl");
        kernel_sources = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        if (kernel_sources.size() == 0)
        {
          throw std::runtime_error("Empty source file! May be you forgot to configure working directory properly?");
        }
        std::cout << kernel_sources << std::endl;
      }

      // TODO 7 Создайте OpenCL-подпрограмму с исходниками кернела
      // см. Runtime APIs -> Program Objects -> clCreateProgramWithSource
      // у string есть метод c_str(), но обратите внимание что передать вам нужно указатель на указатель

      // TODO 8 Теперь скомпилируйте программу и напечатайте в консоль лог компиляции
      // см. clBuildProgram
      program Program{Context, kernel_sources};

      // А так же напечатайте лог компиляции (он будет очень полезен, если в кернеле есть синтаксические ошибки - т.е. когда clBuildProgram вернет CL_BUILD_PROGRAM_FAILURE)
      // Обратите внимание что при компиляции на процессоре через Intel OpenCL драйвер - в логе указывается какой ширины векторизацию получилось выполнить для кернела
      // см. clGetProgramBuildInfo
      //    size_t log_size = 0;
      //    std::vector<char> log(log_size, 0);
      //    if (log_size > 1) {
      //        std::cout << "Log:" << std::endl;
      //        std::cout << log.data() << std::endl;
      //    }
      std::vector<char> BuildLog;

      Program.GetBuildLog(&BuildLog);
      std::cout << "Log:\n" << BuildLog.data() << std::endl;

      // TODO 9 Создайте OpenCL-kernel в созданной подпрограмме (в одной подпрограмме может быть несколько кернелов, но в данном случае кернел один)
      // см. подходящую функцию в Runtime APIs -> Program Objects -> Kernel Objects
      {
        cl_int Error = CL_SUCCESS;

        if ((Error = Program.CreateKernel("aplusb")) != CL_SUCCESS)
        {
          std::cout << "Wrong program, error code: " << Error << std::endl;
          return 30;
        }
      }

      // TODO 10 Выставите все аргументы в кернеле через clSetKernelArg (as_gpu, bs_gpu, cs_gpu и число значений, убедитесь что тип количества элементов такой же в кернеле)
      {
        unsigned int i = 0;
        
        Program.SetArg(i++, sizeof(cl_mem), BufferA.GetBufferPtr());
        Program.SetArg(i++, sizeof(cl_mem), BufferB.GetBufferPtr());
        Program.SetArg(i++, sizeof(cl_mem), BufferC.GetBufferPtr());
        Program.SetArg(i++, sizeof(int), &n);
      }

      // TODO 11 Выше увеличьте n с 1000*1000 до 100*1000*1000 (чтобы дальнейшие замеры были ближе к реальности)

      // TODO 12 Запустите выполнения кернела:
      // - С одномерной рабочей группой размера 128
      // - В одномерном рабочем пространстве размера roundedUpN, где roundedUpN - наименьшее число кратное 128 и при этом не меньшее n
      // - см. clEnqueueNDRangeKernel
      // - Обратите внимание что чтобы дождаться окончания вычислений (чтобы знать когда можно смотреть результаты в cs_gpu) нужно:
      //   - Сохранить событие "кернел запущен" (см. аргумент "cl_event *event")
      //   - Дождаться завершения полунного события - см. в документации подходящий метод среди Event Objects
      {
        size_t workGroupSize = 128;
        size_t global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        timer t; // Это вспомогательный секундомер, он замеряет время своего создания и позволяет усреднять время нескольких замеров
        for (unsigned int i = 0; i < 20; ++i)
        {
          // clEnqueueNDRangeKernel...
          // clWaitForEvents...
          Program.Run1D(Queue, 0, global_work_size, workGroupSize);
          t.nextLap(); // При вызове nextLap секундомер запоминает текущий замер (текущий круг) и начинает замерять время следующего круга
        }
        // Среднее время круга (вычисления кернела) на самом деле считаются не по всем замерам, а лишь с 20%-перцентайля по 80%-перцентайль (как и стандартное отклониение)
        // подробнее об этом - см. timer.lapsFiltered
        // P.S. чтобы в CLion быстро перейти к символу (функции/классу/много чему еще) достаточно нажать Ctrl+Shift+Alt+N -> lapsFiltered -> Enter
        std::cout << "Kernel average time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;

        // TODO 13 Рассчитайте достигнутые гигафлопcы:
        // - Всего элементов в массивах по n штук
        // - Всего выполняется операций: операция a+b выполняется n раз
        // - Флопс - это число операций с плавающей точкой в секунду
        // - В гигафлопсе 10^9 флопсов
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        std::cout << "GFlops: " << n / t.lapAvg() / 1e9 << std::endl;

        // TODO 14 Рассчитайте используемую пропускную способность обращений к видеопамяти (в гигабайтах в секунду)
        // - Всего элементов в массивах по n штук
        // - Размер каждого элемента sizeof(float)=4 байта
        // - Обращений к видеопамяти т.о. 2*n*sizeof(float) байт на чтение и 1*n*sizeof(float) байт на запись, т.е. итого 3*n*sizeof(float) байт
        // - В гигабайте 1024*1024*1024 байт
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        std::cout << "VRAM bandwidth: " << 3.0 * n * sizeof(float) / 1024 / 1024 / 1024 / t.lapAvg() << " GB/s" << std::endl;
      }

      // TODO 15 Скачайте результаты вычислений из видеопамяти (VRAM) в оперативную память (RAM) - из cs_gpu в cs (и рассчитайте скорость трансфера данных в гигабайтах в секунду)
      {
        timer t;
        for (unsigned int i = 0; i < 20; ++i)
        {
          // clEnqueueReadBuffer...
          BufferC.CopyDataFromBuffer(Queue, cs.data(), n * sizeof(float));
          t.nextLap();
        }
        std::cout << "Result data transfer time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "VRAM -> RAM bandwidth: " << n * sizeof(float) / 1024.0 / 1024 / 1024 / t.lapAvg() << " GB/s" << std::endl;
      }

      // TODO 16 Сверьте результаты вычислений со сложением чисел на процессоре (и убедитесь, что если в кернеле сделать намеренную ошибку, то эта проверка поймает ошибку)
      for (unsigned int i = 0; i < n; ++i) {
          if (cs[i] != as[i] + bs[i]) {
              std::cout << cs[i] << " != " << as[i] << " + " << bs[i] << " (i = " << i << ")" << "\n";
              throw std::runtime_error("CPU and GPU results differ!");
          }
      }

    }
  } catch (std::runtime_error & Err)
  {
    std::cout << "Error: " << Err.what() << std::endl;
  }

  return 0;
}
