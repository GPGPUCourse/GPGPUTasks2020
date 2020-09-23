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

template <typename Character>
std::vector<Character> getPlatformInfoString(cl_platform_id platform, cl_platform_info info) {
    size_t stringSize = 0;
    OCL_SAFE_CALL(clGetPlatformInfo(platform, info, 0, nullptr, &stringSize));

    std::vector<Character> clString(stringSize, 0);
    OCL_SAFE_CALL(clGetPlatformInfo(platform, info, stringSize, clString.data(), nullptr));

    return clString;
}

std::vector<cl_device_id> getDevices(cl_platform_id platform, cl_uint* devicesCount) {
    OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, devicesCount));

    std::vector<cl_device_id> devices(*devicesCount);
    OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, *devicesCount, devices.data(), nullptr));

    return devices;
}


template <typename T>
T getDeviceInfoVar(cl_device_id device, cl_device_info info) {
    size_t deviceInfoSize = 0;
    OCL_SAFE_CALL(clGetDeviceInfo(device, info, 0, nullptr, &deviceInfoSize));

    T deviceInfo;
    OCL_SAFE_CALL(clGetDeviceInfo(device, info, deviceInfoSize, &deviceInfo, nullptr));
    return deviceInfo;
}

template <typename Character>
std::vector<Character> getDeviceInfoString(cl_device_id device, cl_device_info info) {
    size_t deviceStringSize = 0;
    OCL_SAFE_CALL(clGetDeviceInfo(device, info, 0, nullptr, &deviceStringSize));

    std::vector<Character> deviceString(deviceStringSize, 0);
    OCL_SAFE_CALL(clGetDeviceInfo(device, info, deviceStringSize, deviceString.data(), nullptr));

    return deviceString;
}


int main()
{
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    cl_uint platformsCount = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
    std::vector<cl_platform_id> platforms(platformsCount);
    OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));

    std::vector<std::vector<cl_device_id>> devices;
    bool deviceFound = false;
    int gpuGroupIndex = -1, gpuDeviceIndex = -1;
    int cpuGroupIndex = -1, cpuDeviceIndex = -1;

    for (int platformIndex = 0; platformIndex < platformsCount; ++platformIndex) {
        cl_platform_id platform = platforms[platformIndex];
        std::cout << "Platform #" << (platformIndex + 1) << "/" << platformsCount << std::endl;
        std::vector<unsigned char> platformName = getPlatformInfoString<unsigned char>(platform, CL_PLATFORM_NAME);
        std::cout << "    Platform name: " << platformName.data() << std::endl;

        cl_uint devicesCount = 0;
        devices.push_back(getDevices(platform, &devicesCount));

        for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
            std::cout << "    Device #" << (deviceIndex + 1) << "/" << devicesCount << std::endl;
            cl_device_id device = devices[platformIndex][deviceIndex];
            std::vector<unsigned char> deviceName = getDeviceInfoString<unsigned char>(device, CL_DEVICE_NAME);
            std::cout << "        Device name: " << deviceName.data() << std::endl;

            auto type = getDeviceInfoVar<cl_device_type>(device, CL_DEVICE_TYPE);
            if (type == CL_DEVICE_TYPE_GPU) {
                deviceFound = true;
                gpuGroupIndex = platformIndex;
                gpuDeviceIndex = deviceIndex;
                break;
            } else if (type == CL_DEVICE_TYPE_CPU) {
                cpuGroupIndex = platformIndex;
                cpuDeviceIndex = deviceIndex;
            }
        }

        if (deviceFound) {
            break;
        }
    }

    int groupIndex, deviceIndex;
    if (gpuGroupIndex != -1) {
        groupIndex = gpuGroupIndex;
        deviceIndex = gpuDeviceIndex;
    } else if (cpuGroupIndex != -1) {
        groupIndex = cpuGroupIndex;
        deviceIndex = cpuDeviceIndex;
    } else {
        throw std::runtime_error("No CPU or GPU devices found");
    }

    cl_int contextError = 0;
    auto selectedDevices = devices[groupIndex];
    cl_context context = clCreateContext(nullptr, selectedDevices.size(), selectedDevices.data(), nullptr, nullptr, &contextError);
    OCL_SAFE_CALL(contextError);

    cl_int commandsError = 0;
    cl_device_id device = selectedDevices[deviceIndex];

    std::vector<unsigned char> deviceName = getDeviceInfoString<unsigned char>(device, CL_DEVICE_NAME);
    std::cout << "\nSelected device name: " << deviceName.data() << std::endl;

    cl_command_queue commandQueue = clCreateCommandQueue(context, device, 0, &commandsError);
    OCL_SAFE_CALL(commandsError);

    unsigned int n = 100 * 1000 * 1000;
    std::vector<float> as(n, 0);
    std::vector<float> bs(n, 0);
    std::vector<float> cs(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.nextf();
        bs[i] = r.nextf();
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;


    cl_int bufferError = 0;
    unsigned int sizeInBytes = n * sizeof(float);

    cl_mem bufferAs = clCreateBuffer(context, CL_MEM_READ_ONLY + CL_MEM_COPY_HOST_PTR,
                                     sizeInBytes, as.data(), &bufferError);
    OCL_SAFE_CALL(bufferError);
    cl_mem bufferBs = clCreateBuffer(context, CL_MEM_READ_ONLY + CL_MEM_COPY_HOST_PTR,
                                     sizeInBytes, bs.data(), &bufferError);
    OCL_SAFE_CALL(bufferError);
    cl_mem bufferCs = clCreateBuffer(context, CL_MEM_WRITE_ONLY + CL_MEM_USE_HOST_PTR,
                                     sizeInBytes, cs.data(), &bufferError);
    OCL_SAFE_CALL(bufferError);
  
    std::string kernel_sources;
    {
        std::ifstream file("src/cl/aplusb.cl");
        kernel_sources = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        if (kernel_sources.size() == 0) {
            throw std::runtime_error("Empty source file! May be you forgot to configure working directory properly?");
        }
        //std::cout << kernel_sources << std::endl;
    }

    cl_int programError;
    const char* cStyleSource = kernel_sources.c_str();
    const size_t length = kernel_sources.size();
    cl_program program = clCreateProgramWithSource(context, 1, &cStyleSource, &length, &programError);
    OCL_SAFE_CALL(programError);


    OCL_SAFE_CALL(clBuildProgram(program, selectedDevices.size(), selectedDevices.data(),
                                 nullptr, nullptr, nullptr));

    size_t log_size = 0;
    OCL_SAFE_CALL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size));
    std::vector<char> log(log_size, 0);
    OCL_SAFE_CALL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr));

    if (log_size > 1) {
        std::cout << "Log:" << std::endl;
        std::cout << log.data() << std::endl;
    }


    cl_int kernelError = 0;
    cl_kernel kernel = clCreateKernel(program, "aplusb", &kernelError);
    OCL_SAFE_CALL(kernelError);



    {
        unsigned int i = 0;
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(cl_mem), &bufferAs));
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(cl_mem), &bufferBs));
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(cl_mem), &bufferCs));
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(unsigned int), &n));
    }


    const uint32_t gig = 1024 * 1024 * 1024;
    {
        cl_event flopsEvent;
        size_t workGroupSize = 128;
        size_t global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        timer t; // Это вспомогательный секундомер, он замеряет время своего создания и позволяет усреднять время нескольких замеров
        for (unsigned int i = 0; i < 20; ++i) {
            OCL_SAFE_CALL(clEnqueueNDRangeKernel(commandQueue, kernel, 1, nullptr, &global_work_size,
                                                 &workGroupSize, 0, nullptr, &flopsEvent));
            OCL_SAFE_CALL(clWaitForEvents(1, &flopsEvent));
            t.nextLap(); // При вызове nextLap секундомер запоминает текущий замер (текущий круг) и начинает замерять время следующего круга
        }
        std::cout << "Kernel average time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;

        std::cout << "GFlops: " << n / t.lapAvg() / 1e9 << std::endl;

        std::cout << "VRAM bandwidth: " << 3 * n * sizeof(float) / t.lapAvg() / gig << " GB/s" << std::endl;
        OCL_SAFE_CALL(clReleaseEvent(flopsEvent));
    }

    {
        timer t;
        cl_event transferEvent;
        for (unsigned int i = 0; i < 20; ++i) {
            OCL_SAFE_CALL(clEnqueueReadBuffer(commandQueue, bufferCs, CL_FALSE, 0, sizeof(float) * n,
                                              cs.data(), 0, nullptr, &transferEvent));
            OCL_SAFE_CALL(clWaitForEvents(1, &transferEvent));
            t.nextLap();
        }
        std::cout << "Result data transfer time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "VRAM -> RAM bandwidth: " << n * sizeof(float) / t.lapAvg() / gig << " GB/s" << std::endl;
        OCL_SAFE_CALL(clReleaseEvent(transferEvent));
    }

    for (unsigned int i = 0; i < n; ++i) {
        if (cs[i] != as[i] + bs[i]) {
            throw std::runtime_error("CPU and GPU results differ!");
        }
    }

    OCL_SAFE_CALL(clReleaseKernel(kernel));
    OCL_SAFE_CALL(clReleaseProgram(program));
    OCL_SAFE_CALL(clReleaseMemObject(bufferCs));
    OCL_SAFE_CALL(clReleaseMemObject(bufferBs));
    OCL_SAFE_CALL(clReleaseMemObject(bufferAs));
    OCL_SAFE_CALL(clReleaseCommandQueue(commandQueue));
    OCL_SAFE_CALL(clReleaseContext(context));

    return 0;
}
