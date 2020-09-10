#include <CL/cl.h>
#include <libclew/ocl_init.h>

#include <vector>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <tuple>


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

template<class T = char, class P = size_t, class... Ps, class... Rs>
std::pair<std::vector<T>, P> extractFromFunc(cl_int (*clFunc)(Ps...), Rs... args)
{
    P paramName = 0;
    OCL_SAFE_CALL(clFunc(args..., 0, nullptr, &paramName));
    std::vector<T> returnVec(paramName, (T) 0);
    OCL_SAFE_CALL(clFunc(args..., paramName, returnVec.data(), nullptr));
    return std::make_pair(returnVec, paramName);
}

std::string getDeviceTypeName(cl_device_type type) {
    switch (type) {
        case CL_DEVICE_TYPE_DEFAULT:
            return "CL_DEVICE_TYPE_DEFAULT";
        case CL_DEVICE_TYPE_CPU:
            return "CL_DEVICE_TYPE_CPU";
        case CL_DEVICE_TYPE_GPU:
            return "CL_DEVICE_TYPE_GPU";
        case CL_DEVICE_TYPE_ACCELERATOR:
            return "CL_DEVICE_TYPE_ACCELERATOR";
        default:
            return "CL_DEVICE_TYPE_ALL";
    }
}

int main()
{
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    cl_uint platformsCount = 0;
    std::vector<cl_platform_id> platforms;
    std::tie(platforms, platformsCount) = extractFromFunc<cl_platform_id, cl_uint>(&clGetPlatformIDs);
    std::cout << "Number of OpenCL platforms: " << platformsCount << std::endl;

    for (int platformIndex = 0; platformIndex < platformsCount; ++platformIndex) {
        std::cout << "Platform #" << (platformIndex + 1) << "/" << platformsCount << std::endl;
        cl_platform_id platform = platforms[platformIndex];

        std::string platformName   = extractFromFunc(&clGetPlatformInfo, platform, CL_PLATFORM_NAME).first.data();
        std::string platformVendor = extractFromFunc(&clGetPlatformInfo, platform, CL_PLATFORM_VENDOR).first.data();
        std::cout << "    Platform name: "   << platformName.data()   << std::endl;
        std::cout << "    Platform vendor: " << platformVendor.data() << std::endl;

        cl_uint devicesCount = 0;
        std::vector<cl_device_id> devices;
        std::tie(devices, devicesCount) = extractFromFunc<cl_device_id, cl_uint>(&clGetDeviceIDs, platform, CL_DEVICE_TYPE_ALL);

        for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
            std::cout << "    Device #" << (deviceIndex + 1) << "/" << devicesCount << std::endl;

            std::string    deviceName      = extractFromFunc                (&clGetDeviceInfo, devices[deviceIndex], CL_DEVICE_NAME).first.data();
            cl_device_type deviceType      = extractFromFunc<cl_device_type>(&clGetDeviceInfo, devices[deviceIndex], CL_DEVICE_TYPE).first[0];
            cl_ulong       deviceMemSize   = extractFromFunc<cl_ulong>      (&clGetDeviceInfo, devices[deviceIndex], CL_DEVICE_GLOBAL_MEM_SIZE).first[0];
            cl_bool        deviceAvailable = extractFromFunc<cl_bool>       (&clGetDeviceInfo, devices[deviceIndex], CL_DEVICE_AVAILABLE).first[0];
            cl_uint        deviceMaxClocks = extractFromFunc<cl_uint>       (&clGetDeviceInfo, devices[deviceIndex], CL_DEVICE_MAX_CLOCK_FREQUENCY).first[0];

            std::cout << "        Device name: "           << deviceName.data()                      << std::endl;
            std::cout << "        Device type: "           << getDeviceTypeName(deviceType)          << std::endl;
            std::cout << "        Device mem size: "       << deviceMemSize / (1024 * 1024) << " MB" << std::endl;
            std::cout << "        Device is available: "   << (deviceAvailable ? "true" : "false")   << std::endl;
            std::cout << "        Device max clock freq: " << deviceMaxClocks << " Hz"               << std::endl;
        }
    }

    return 0;
}