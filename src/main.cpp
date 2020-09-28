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


template<typename T>
std::string to_string(T value) {
    std::ostringstream ss;
    ss << value;
    return ss.str();
}

void reportError(cl_int err, const std::string &filename, int line) {
    if (CL_SUCCESS == err)
        return;

    // Таблица с кодами ошибок:
    // libs/clew/CL/cl.h:103
    // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
    std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
    throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)


int main() {
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку clew)
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    // TODO 1 По аналогии с предыдущим заданием узнайте, какие есть устройства, и выберите из них какое-нибудь
    cl_platform_id chosenPlatform = nullptr;
    cl_device_id chosenDevice = nullptr;

    cl_uint platformsCount = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));

    std::vector<cl_platform_id> platforms(platformsCount);
    OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));

    for (int platformIndex = 0; platformIndex < platformsCount; ++platformIndex) {
        cl_platform_id platform = platforms[platformIndex];
        size_t platformNameSize = 0;
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameSize));
        std::vector<unsigned char> platformName(platformNameSize, 0);
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr));

        std::string platformNameString = std::string(
                reinterpret_cast<const char *>(&platformName[0]),
                platformName.size());

        cl_uint devicesCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));

        std::vector<cl_device_id> devices(devicesCount);
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));

        for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
            cl_device_id device = devices[deviceIndex];

            cl_device_type deviceType{};
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &deviceType, nullptr));
            if (deviceType == CL_DEVICE_TYPE_GPU) {
                chosenDevice = device;
                chosenPlatform = platform;
            } else if (deviceType == CL_DEVICE_TYPE_CPU) {
                chosenDevice = device;
                chosenPlatform = platform;
            } else {
                throw std::runtime_error("No CPU or GPU devices!");
            }
        }

        // I'd prefere NVIDIA devices
        if (platformNameString == std::string("NVIDIA CUDA")) {
            break;
        }
    }

    // TODO 2 Создайте контекст с выбранным устройством
    cl_int errorCode = 0;
    cl_context context = clCreateContext(nullptr, 1, &chosenDevice, nullptr, nullptr, &errorCode);
    OCL_SAFE_CALL(errorCode);

    // TODO 3 Создайте очередь выполняемых команд в рамках выбранного контекста и устройства
    cl_command_queue clCommandQueue = clCreateCommandQueue(context, chosenDevice, 0, &errorCode);
    OCL_SAFE_CALL(errorCode);

    unsigned int n = 100 * 1000 * 1000;
    // Создаем два массива псевдослучайных данных для сложения и массив для будущего хранения результата
    std::vector<float> as(n, 0);
    std::vector<float> bs(n, 0);
    std::vector<float> cs(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.nextf();
        bs[i] = r.nextf();
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    // TODO 4 Создайте три буфера в памяти устройства (в случае видеокарты - в видеопамяти - VRAM) - для двух суммируемых массивов as и bs (они read-only) и для массива с результатом cs (он write-only)
    unsigned long bufferSize = sizeof(float) * n;

    cl_mem aBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, bufferSize, nullptr, &errorCode);
    OCL_SAFE_CALL(errorCode);

    cl_mem bBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, bufferSize, nullptr, &errorCode);
    OCL_SAFE_CALL(errorCode);

    cl_mem cBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bufferSize, nullptr, &errorCode);
    OCL_SAFE_CALL(errorCode);

    OCL_SAFE_CALL(
            clEnqueueWriteBuffer(clCommandQueue, aBuffer, CL_TRUE, 0, bufferSize, as.data(), 0, nullptr, nullptr));
    OCL_SAFE_CALL(
            clEnqueueWriteBuffer(clCommandQueue, bBuffer, CL_TRUE, 0, bufferSize, bs.data(), 0, nullptr, nullptr));

    // TODO 6 Выполните TODO 5 (реализуйте кернел в src/cl/aplusb.cl)
    std::string kernel_sources;
    {
        std::ifstream file("src/cl/aplusb.cl");
        kernel_sources = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        if (kernel_sources.size() == 0) {
            throw std::runtime_error("Empty source file! May be you forgot to configure working directory properly?");
        }
        std::cout << kernel_sources << std::endl;
    }

    // TODO 7 Создайте OpenCL-подпрограмму с исходниками кернела
    const char *sourcePtr = kernel_sources.c_str();
    cl_program program = clCreateProgramWithSource(context, 1, &sourcePtr, nullptr, &errorCode);
    OCL_SAFE_CALL(errorCode);

    // TODO 8 Теперь скомпилируйте программу и напечатайте в консоль лог компиляции
    OCL_SAFE_CALL(clBuildProgram(program, 1, &chosenDevice, nullptr, nullptr, nullptr));

    size_t log_size = 0;
    OCL_SAFE_CALL(clGetProgramBuildInfo(program, chosenDevice, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size));
    std::vector<char> log(log_size, 0);
    OCL_SAFE_CALL(clGetProgramBuildInfo(program, chosenDevice, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr));
    if (log_size > 1) {
        std::cout << "Log:" << std::endl;
        std::cout << log.data() << std::endl;
    }
    OCL_SAFE_CALL(errorCode);

    // TODO 9 Создайте OpenCL-kernel в созданной подпрограмме (в одной подпрограмме может быть несколько кернелов, но в данном случае кернел один)
    auto kernel = clCreateKernel(program, "aplusb", &errorCode);
    OCL_SAFE_CALL(errorCode);

    // TODO 10 Выставите все аргументы в кернеле через clSetKernelArg (as_gpu, bs_gpu, cs_gpu и число значений, убедитесь, что тип количества элементов такой же в кернеле)
    {
        unsigned int i = 0;
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(cl_mem), &aBuffer));
        OCL_SAFE_CALL(errorCode);
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(cl_mem), &bBuffer));
        OCL_SAFE_CALL(errorCode);
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(cl_mem), &cBuffer));
        OCL_SAFE_CALL(errorCode);
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(unsigned int), &n));
        OCL_SAFE_CALL(errorCode);
    }

    // TODO 11 Выше увеличьте n с 1000*1000 до 100*1000*1000 (чтобы дальнейшие замеры были ближе к реальности)

    // TODO 12 Запустите выполнения кернела:
    {
        size_t workGroupSize = 128;
        size_t global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        timer t;
        for (unsigned int i = 0; i < 20; ++i) {
            cl_event clEvent;
            OCL_SAFE_CALL(
                    clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, nullptr, &global_work_size, &workGroupSize, 0,
                                           nullptr, &clEvent));
            OCL_SAFE_CALL(clWaitForEvents(1, &clEvent));
            OCL_SAFE_CALL(clReleaseEvent(clEvent));
            t.nextLap();
        }
        std::cout << "Kernel average time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;

        // TODO 13 Рассчитайте достигнутые гигафлопcы:
        std::cout << "GFlops: " << n / t.lapAvg() / 1e9 << std::endl;

        // TODO 14 Рассчитайте используемую пропускную способность обращений к видеопамяти (в гигабайтах в секунду)
        std::cout << "VRAM bandwidth: " << 3 * bufferSize / t.lapAvg() / (1024 * 1024 * 1024) << " GB/s" << std::endl;
    }

    // TODO 15 Скачайте результаты вычислений из видеопамяти (VRAM) в оперативную память (RAM) - из cs_gpu в cs (и рассчитайте скорость трансфера данных в гигабайтах в секунду)
    {
        timer t;
        for (unsigned int i = 0; i < 20; ++i) {
            OCL_SAFE_CALL(clEnqueueReadBuffer(clCommandQueue, cBuffer, CL_TRUE, 0, bufferSize, cs.data(), 0, nullptr,
                                              nullptr));
            t.nextLap();
        }
        std::cout << "Result data transfer time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "VRAM -> RAM bandwidth: " << bufferSize / t.lapAvg() / (1024 * 1024 * 1024) << " GB/s"
                  << std::endl;
    }

    // TODO 16 Сверьте результаты вычислений со сложением чисел на процессоре (и убедитесь, что если в кернеле сделать намеренную ошибку, то эта проверка поймает ошибку)
    for (unsigned int i = 0; i < n; ++i) {
        if (cs[i] != as[i] + bs[i]) {
            throw std::runtime_error("CPU and GPU results differ!");
        }
    }

    OCL_SAFE_CALL(clReleaseProgram(program));
    OCL_SAFE_CALL(clReleaseKernel(kernel));
    OCL_SAFE_CALL(clReleaseContext(context));
    OCL_SAFE_CALL(clReleaseCommandQueue(clCommandQueue));
    OCL_SAFE_CALL(clReleaseMemObject(aBuffer));
    OCL_SAFE_CALL(clReleaseMemObject(bBuffer));
    OCL_SAFE_CALL(clReleaseMemObject(cBuffer));

    return 0;
}
