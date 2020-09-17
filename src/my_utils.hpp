#include <CL/cl.h>
#include <libclew/ocl_init.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>

#include <functional>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

inline void oclInitIfNeeded() {
    static std::optional<bool> isInitialised = false;

    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку libs/clew)
    if (!isInitialised)
        isInitialised = ocl_init();
    
    if (!isInitialised.value())
        throw std::runtime_error("Can't init OpenCL driver!");
}

template <typename T>
inline std::string to_string(T value)
{
    std::ostringstream ss;
    ss << value;
    return ss.str();
}

inline void reportError(cl_int err, const std::string &filename, int line)
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

template<cl_device_info info, typename ProcessFunc>
inline void printDeviceInfo(const cl::Device &device, std::string description, const ProcessFunc &processFunc) {
    cl_int result;
    const auto resultValue = device.getInfo<info>(&result);
    OCL_SAFE_CALL(result);
    std::cout << "        " << std::move(description) << ": " << processFunc(resultValue) << std::endl;
}

template<cl_device_info info>
inline void printDeviceInfo(const cl::Device &device, std::string description) {
    using T = decltype(device.getInfo<info>(nullptr));
    printDeviceInfo<info>(device, description, [](T value) { return value; });
}

void printOpenCLDevices() {
    oclInitIfNeeded();

    std::vector<cl::Platform> platforms;
    OCL_SAFE_CALL(cl::Platform::get(&platforms));

    std::cout << "Number of OpenCL platforms: " << platforms.size() << std::endl;

    for (size_t platformIndex = 0; platformIndex < platforms.size(); ++platformIndex) {
        std::cout << "Platform #" << (platformIndex + 1) << "/" << platforms.size() << ":" << std::endl;
        const auto &platform = platforms.at(platformIndex);
        
        const auto printPlatformInfo = [&platform](std::string description, cl_platform_info info) {
            static std::string platformInfoString;
            OCL_SAFE_CALL(platform.getInfo(info, &platformInfoString));
            std::cout << "    " << description << ": " << platformInfoString << std::endl;
        };
        
        printPlatformInfo("Name", CL_PLATFORM_NAME);

        printPlatformInfo("Vendor", CL_PLATFORM_VENDOR);

        std::vector<cl::Device> devices;
        OCL_SAFE_CALL(platform.getDevices(CL_DEVICE_TYPE_ALL, &devices));
        
        for (size_t deviceIndex = 0; deviceIndex < devices.size(); ++deviceIndex) {
            std::cout << "    Device #" << (deviceIndex + 1) << "/" << devices.size() << ":" << std::endl;
            const auto &device = devices.at(deviceIndex);
        
            printDeviceInfo<CL_DEVICE_NAME>(device, "Name");
            
            printDeviceInfo<CL_DEVICE_TYPE>(device, "Type", [](cl_device_type type) {
                switch (type) {
                    case CL_DEVICE_TYPE_CPU:
                        return "CPU";
                    case CL_DEVICE_TYPE_GPU:
                        return "GPU";
                    default:
                        return "unknown";
                }
            });
            
            printDeviceInfo<CL_DEVICE_GLOBAL_MEM_SIZE>(device, "Total memory available (MB)", [](size_t bytes) { return bytes >> 20; });
            
            printDeviceInfo<CL_DEVICE_MAX_COMPUTE_UNITS>(device, "CUs");

            printDeviceInfo<CL_DEVICE_OPENCL_C_VERSION>(device, "OpenCL version");
        }
    }
}

template<cl_device_type type>
inline std::optional<cl::Device> getAnyDeviceByType() {
    oclInitIfNeeded();

    std::vector<cl::Platform> platforms;
    OCL_SAFE_CALL(cl::Platform::get(&platforms));

    for (const auto &platform : platforms) {
        std::vector<cl::Device> devices;
        OCL_SAFE_CALL(platform.getDevices(type, &devices));
        
        for (const auto &device : devices)
            return device;
    }

    return {};
}

inline cl::Device getSuitableDevice() {
    oclInitIfNeeded();

    const auto firstGPU = getAnyDeviceByType<CL_DEVICE_TYPE_GPU>();
    if (firstGPU)
        return firstGPU.value();

    const auto firstCPU = getAnyDeviceByType<CL_DEVICE_TYPE_CPU>();
    if (firstCPU)
        return firstCPU.value();

    // that's some strange PC
    OCL_SAFE_CALL(CL_DEVICE_NOT_FOUND);
}