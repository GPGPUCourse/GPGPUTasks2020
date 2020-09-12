#include <CL/cl.h>
#include <libclew/ocl_init.h>

#include <vector>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <math.h>

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

template<typename Entity>
std::vector<unsigned char> cl_entity_info(Entity entity,
                    cl_uint info,
                    cl_int info_func(Entity, cl_uint, size_t, void *, size_t *)) {

    size_t paramValueSize = 0;

    OCL_SAFE_CALL(info_func(entity, info, 0, nullptr, &paramValueSize));
    std::vector<unsigned char> paramValue(paramValueSize, 0);
    OCL_SAFE_CALL(info_func(entity, info, paramValueSize, paramValue.data(), nullptr));
    return paramValue;
}

template<typename Result, typename Entity>
Result cl_entity_info(Entity entity,
                      cl_uint info,
                      cl_int info_func(Entity, cl_uint, size_t, void *, size_t *)) {

    size_t paramValueSize = 0;

    OCL_SAFE_CALL(info_func(entity, info, 0, nullptr, &paramValueSize));
    Result paramValue;
    OCL_SAFE_CALL(info_func(entity, info, paramValueSize, &paramValue, nullptr));
    return paramValue;
}

int main()
{
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку libs/clew)
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    // Откройте 
    // https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/
    // Нажмите слева: "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformIDs"
    // Прочитайте документацию clGetPlatformIDs и убедитесь что этот способ узнать сколько есть платформ соответствует документации:
    cl_uint platformsCount = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
    std::cout << "Number of OpenCL platforms: " << platformsCount << std::endl;

    std::vector<cl_platform_id> platforms {platformsCount};
    OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));

    for (auto const& platform: platforms) {

        std::vector<unsigned char> platformName = cl_entity_info(platform, CL_PLATFORM_NAME, clGetPlatformInfo);
        std::cout << "\tPlatform name: " << platformName.data() << std::endl;

        std::vector<unsigned char> vendorName = cl_entity_info(platform, CL_PLATFORM_VENDOR, clGetPlatformInfo);
        std::cout << "\tVendor name: " << vendorName.data() << std::endl;

        cl_uint devicesCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));

        std::vector<cl_device_id> devices {devicesCount};
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));

        for(auto const& device: devices) {
            auto deviceName = cl_entity_info(device, CL_DEVICE_NAME, clGetDeviceInfo);
            std::cout << "\t\tDevice name: " << deviceName.data() << std::endl;
            auto deviceType = cl_entity_info<cl_device_type>(device, CL_DEVICE_TYPE, clGetDeviceInfo);
            std::cout << "\t\tDevice type: ";
            switch (deviceType) {
                case CL_DEVICE_TYPE_GPU:
                    std::cout << " GPU" ;
                    break;
                case CL_DEVICE_TYPE_CPU:
                    std::cout << " CPU" ;
                    break;
                default:
                    break;
            }
            std::cout << std::endl;

            auto deviceMemory = cl_entity_info<cl_ulong>(device, CL_DEVICE_GLOBAL_MEM_SIZE, clGetDeviceInfo);

            std::cout << "\t\tDevice memory size: " << round(deviceMemory / (1024.0 * 1024.0)) << " MB" << std::endl;
            std::cout << "—————————————" << std::endl;
        }
    }
    return 0;
}