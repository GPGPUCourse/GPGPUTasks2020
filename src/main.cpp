#include <CL/cl.h>
#include <libclew/ocl_init.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>

#include <vector>
#include <unordered_map>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <cassert>


typedef std::vector<unsigned char> vchar;

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

void getPlatformIDs(cl_uint& platformsCount, cl_platform_id* platforms) {
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
    OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms, nullptr));
}

template<class T>
void getPlatformProperty(cl_platform_id platform, cl_platform_info param_name, T* property) {
    size_t propertySize = 0;
    OCL_SAFE_CALL(clGetPlatformInfo(platform, param_name, 0, nullptr, &propertySize));
    OCL_SAFE_CALL(clGetPlatformInfo(platform, param_name, propertySize, property, nullptr));
}

void getDeviceIDs(cl_platform_id platform, cl_device_type device_type, cl_uint& devicesCount, cl_device_id* devices) {
    OCL_SAFE_CALL(clGetDeviceIDs(platform, device_type, 0, nullptr, &devicesCount));
    OCL_SAFE_CALL(clGetDeviceIDs(platform, device_type, devicesCount, devices, nullptr));
}

template<class T>
void getDeviceProperty(cl_device_id device, cl_device_info param_name, T* property) {
    size_t propertySize = 0;
    OCL_SAFE_CALL(clGetDeviceInfo(device, param_name, 0, nullptr, &propertySize));
    OCL_SAFE_CALL(clGetDeviceInfo(device, param_name, propertySize, property, nullptr));
}

std::string getDeviceTypeName(cl_device_type deviceType) {
    static std::unordered_map<uint64_t, std::string> typeToString = {
            {CL_DEVICE_TYPE_DEFAULT, "DEFAULT"},
            {CL_DEVICE_TYPE_CPU, "CPU"},
            {CL_DEVICE_TYPE_GPU, "GPU"},
            {CL_DEVICE_TYPE_ACCELERATOR, "ACCELERATOR"},
            {CL_DEVICE_TYPE_ALL, "ALL"}
    };
    return typeToString[deviceType];
}

void selectDevice(cl_device_id& workDevice, cl_device_type& workDeviceType) {
    cl_uint platformsCount = 0;
    std::vector<cl_platform_id> platforms(8);
    getPlatformIDs(platformsCount, platforms.data());

    for (cl_uint platformIndex = 0; platformIndex < platformsCount; ++platformIndex) {
        std::cout << "Platform #" << (platformIndex + 1) << "/" << platformsCount << std::endl;
        cl_platform_id platform = platforms[platformIndex];

        vchar platformName(256, 0);
        getPlatformProperty(platform, CL_PLATFORM_NAME, platformName.data());

        cl_uint devicesCount = 0;
        std::vector<cl_device_id> devices(8);
        getDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data());

        std::cout << "    Platform name: " << platformName.data() << std::endl;
        std::cout << "    Platform devices count: " << devicesCount << std::endl;

        for (cl_uint deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
            std::cout << "Device #" << (deviceIndex + 1) << "/" << devicesCount << std::endl;
            cl_device_id device = devices[deviceIndex];

            vchar deviceName(256, 0);
            getDeviceProperty(device, CL_DEVICE_NAME, deviceName.data());
            cl_device_type deviceType;
            getDeviceProperty(device, CL_DEVICE_TYPE, &deviceType);

            if (deviceIndex == 0 || deviceType == CL_DEVICE_TYPE_GPU) {
                workDevice = device;
                workDeviceType = deviceType;
            }

            std::cout << "    device name: " << deviceName.data() << std::endl;
            std::cout << "    device type: " << getDeviceTypeName(deviceType) << std::endl;
        }
    }

    std::cout << "Selected work device type: " << getDeviceTypeName(workDeviceType) << std::endl;
}

int main()
{    
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку clew)
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    // TODO 1 По аналогии с предыдущим заданием узнайте какие есть устройства, и выберите из них какое-нибудь
    // (если в списке устройств есть хоть одна видеокарта - выберите ее, если нету - выбирайте процессор)
    cl_device_id workDevice;
    cl_device_type workDeviceType = 0;
    selectDevice(workDevice, workDeviceType);

    // TODO 2 Создайте контекст с выбранным устройством
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Contexts -> clCreateContext
    // Не забывайте проверять все возвращаемые коды на успешность (обратите внимание что в данном случае метод возвращает
    // код по переданному аргументом errcode_ret указателю)
    // И хорошо бы сразу добавить в конце clReleaseContext (да, не очень RAII, но это лишь пример)
    cl_int errcode_ret;
    cl_context context = clCreateContext(
            nullptr, // properties
            1, // num_devices
            &workDevice, // devices,
            nullptr, // callback
            nullptr, // user_data
            &errcode_ret
    );
    OCL_SAFE_CALL(errcode_ret);

    // TODO 3 Создайте очередь выполняемых команд в рамках выбранного контекста и устройства
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Runtime APIs -> Command Queues -> clCreateCommandQueue
    // Убедитесь что в соответствии с документацией вы создали in-order очередь задач
    // И хорошо бы сразу добавить в конце clReleaseCommandQueue (не забывайте освобождать ресурсы)
    cl_command_queue command_queue = clCreateCommandQueue(
            context,
            workDevice,
            0, // properties
            &errcode_ret
    );
    OCL_SAFE_CALL(errcode_ret);

    unsigned int n = 100*1000*1000;
    // Создаем два массива псевдослучайных данных для сложения и массив для будущего хранения результата
    std::vector<float> as(n, 0);
    std::vector<float> bs(n, 0);
    std::vector<float> cs(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.nextf();
        bs[i] = r.nextf();
    }
    std::cout << "Data generated for n=" << n  << std::endl;

    // TODO 4 Создайте три буфера в памяти устройства (в случае видеокарты - в видеопамяти - VRAM) - для двух суммируемых массивов as и bs (они read-only) и для массива с результатом cs (он write-only)
    // См. Buffer Objects -> clCreateBuffer
    // Размер в байтах соответственно можно вычислить через sizeof(float)=4 и тот факт что чисел в каждом массиве - n штук
    // Данные в as и bs можно прогрузить этим же методом скопировав данные из host_ptr=as.data() (и не забыв про битовый флаг на это указывающий)
    // или же через метод Buffer Objects -> clEnqueueWriteBuffer
    // И хорошо бы сразу добавить в конце clReleaseMemObject (аналогично все дальнейшие ресурсы вроде OpenCL под-программы, кернела и т.п. тоже нужно освобождать)
    std::size_t buffer_size = n * sizeof(float);
    cl_mem_flags flags = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;

    cl_mem as_buffer = clCreateBuffer(
            context,
            flags,
            buffer_size,
            as.data(), // host_ptr
            &errcode_ret
    );
    OCL_SAFE_CALL(errcode_ret);

    cl_mem bs_buffer = clCreateBuffer(
            context,
            flags,
            buffer_size,
            bs.data(), // host_ptr
            &errcode_ret
    );
    OCL_SAFE_CALL(errcode_ret);

    cl_mem cs_buffer = clCreateBuffer(
            context,
            flags,
            buffer_size,
            cs.data(), // host_ptr
            &errcode_ret
    );
    OCL_SAFE_CALL(errcode_ret);

    // TODO 6 Выполните TODO 5 (реализуйте кернел в src/cl/aplusb.cl)
    // затем убедитесь что выходит загрузить его с диска (убедитесь что Working directory выставлена правильно - см. описание задания)
    // напечатав исходники в консоль (if проверяет что удалось считать хоть что-то)
    std::string kernel_sources;
    {
        std::ifstream file("src/cl/aplusb.cl");
        kernel_sources = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        if (kernel_sources.empty()) {
            throw std::runtime_error("Empty source file! May be you forgot to configure working directory properly?");
        }
        std::cout << kernel_sources << std::endl;
    }

    // TODO 7 Создайте OpenCL-подпрограмму с исходниками кернела
    // см. Runtime APIs -> Program Objects -> clCreateProgramWithSource
    // у string есть метод c_str(), но обратите внимание что передать вам нужно указатель на указатель
    const char* kernel_sources_c_str = kernel_sources.c_str();
    cl_program program = clCreateProgramWithSource(
            context,
            1, // count
            &kernel_sources_c_str, // strings
            nullptr, // lengths
            &errcode_ret
    );
    OCL_SAFE_CALL(errcode_ret);
    
    // TODO 8 Теперь скомпилируйте программу и напечатайте в консоль лог компиляции
    // см. clBuildProgram
    OCL_SAFE_CALL(clBuildProgram(
            program,
            0, // num_devices
            nullptr, // device_list (if NULL, the program executable is built for all devices associated with program)
            nullptr, // options
            nullptr, // pfn_notify()
            nullptr // user_data
    ));

    // А так же напечатайте лог компиляции (он будет очень полезен, если в кернеле есть синтаксические ошибки - т.е. когда clBuildProgram вернет CL_BUILD_PROGRAM_FAILURE)
    // Обратите внимание что при компиляции на процессоре через Intel OpenCL драйвер - в логе указывается какой ширины векторизацию получилось выполнить для кернела
    // см. clGetProgramBuildInfo
    size_t log_size = 0;
    OCL_SAFE_CALL(clGetProgramBuildInfo(
            program,
            workDevice,
            CL_PROGRAM_BUILD_LOG, // param_name
            0, // param_value_size
            nullptr, // param_value
            &log_size // param_value_size_ret
    ));

    std::vector<char> log(log_size, 0);
    OCL_SAFE_CALL(clGetProgramBuildInfo(
            program,
            workDevice,
            CL_PROGRAM_BUILD_LOG, // param_name
            log_size, // param_value_size
            log.data(), // param_value
            nullptr // param_value_size_ret
    ));

    if (!log.empty()) {
        std::cout << "Log:\n" << log.data() << std::endl;
    }

    // TODO 9 Создайте OpenCL-kernel в созданной подпрограмме (в одной подпрограмме может быть несколько кернелов, но в данном случае кернел один)
    // см. подходящую функцию в Runtime APIs -> Program Objects -> Kernel Objects
    cl_kernel aplusb_kernel = clCreateKernel(
            program,
            "aplusb", //kernel_name,
            &errcode_ret
    );
    OCL_SAFE_CALL(errcode_ret);

    // TODO 10 Выставите все аргументы в кернеле через clSetKernelArg (as_gpu, bs_gpu, cs_gpu и число значений, убедитесь что тип количества элементов такой же в кернеле)
    OCL_SAFE_CALL(clSetKernelArg(aplusb_kernel, 0, buffer_size, &as_buffer));
    OCL_SAFE_CALL(clSetKernelArg(aplusb_kernel, 1, buffer_size, &bs_buffer));
    OCL_SAFE_CALL(clSetKernelArg(aplusb_kernel, 2, buffer_size, &cs_buffer));
    OCL_SAFE_CALL(clSetKernelArg(aplusb_kernel, 3, sizeof(unsigned), &n));

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
        for (unsigned int i = 0; i < 20; ++i) {
            // clEnqueueNDRangeKernel...
            // clWaitForEvents...
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
        std::cout << "GFlops: " << 0 << std::endl;

        // TODO 14 Рассчитайте используемую пропускную способность обращений к видеопамяти (в гигабайтах в секунду)
        // - Всего элементов в массивах по n штук
        // - Размер каждого элемента sizeof(float)=4 байта
        // - Обращений к видеопамяти т.о. 2*n*sizeof(float) байт на чтение и 1*n*sizeof(float) байт на запись, т.е. итого 3*n*sizeof(float) байт
        // - В гигабайте 1024*1024*1024 байт
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        std::cout << "VRAM bandwidth: " << 0 << " GB/s" << std::endl;
    }

    // TODO 15 Скачайте результаты вычислений из видеопамяти (VRAM) в оперативную память (RAM) - из cs_gpu в cs (и рассчитайте скорость трансфера данных в гигабайтах в секунду)
    {
        timer t;
        for (unsigned int i = 0; i < 20; ++i) {
            // clEnqueueReadBuffer...
            t.nextLap();
        }
        std::cout << "Result data transfer time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "VRAM -> RAM bandwidth: " << 0 << " GB/s" << std::endl;
    }

    // TODO 16 Сверьте результаты вычислений со сложением чисел на процессоре (и убедитесь, что если в кернеле сделать намеренную ошибку, то эта проверка поймает ошибку)
//    for (unsigned int i = 0; i < n; ++i) {
//        if (cs[i] != as[i] + bs[i]) {
//            throw std::runtime_error("CPU and GPU results differ!");
//        }
//    }

    OCL_SAFE_CALL(clReleaseKernel(aplusb_kernel));
    OCL_SAFE_CALL(clReleaseProgram(program));
    OCL_SAFE_CALL(clReleaseMemObject(cs_buffer));
    OCL_SAFE_CALL(clReleaseMemObject(bs_buffer));
    OCL_SAFE_CALL(clReleaseMemObject(as_buffer));
    OCL_SAFE_CALL(clReleaseCommandQueue(command_queue));
    OCL_SAFE_CALL(clReleaseContext(context));

    return 0;
}
