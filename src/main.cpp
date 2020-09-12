#include <CL/cl.h>
#include <libclew/ocl_init.h>

#include <vector>
#include <sstream>
#include <iostream>
#include <stdexcept>


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

    // Тот же метод используется для того чтобы получить идентификаторы всех платформ - сверьтесь с документацией, что это сделано верно:
    std::vector<cl_platform_id> platforms(platformsCount);
    OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));

    for (int platformIndex = 0; platformIndex < platformsCount; ++platformIndex) {
        std::cout << "Platform #" << (platformIndex + 1) << "/" << platformsCount << std::endl;
        cl_platform_id platform = platforms[platformIndex];

        // Откройте документацию по "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformInfo"
        // Не забывайте проверять коды ошибок с помощью макроса OCL_SAFE_CALL
        size_t platformNameSize = 0;
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameSize));

        std::vector<unsigned char> platformName(platformNameSize, 0);
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr));
        std::cout << "    Platform name: " << platformName.data() << std::endl;

        size_t platformVendorSize = 0;
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &platformVendorSize));
        std::vector<unsigned char> platformVendor(platformVendorSize, 0);
        OCL_SAFE_CALL(
                clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, platformVendorSize, platformVendor.data(), nullptr));
        std::cout << "    Platform vendor: " << platformVendor.data() << std::endl;

        cl_uint devicesCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
        std::vector<cl_device_id> devices(devicesCount);
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));
        for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
            std::cout << "    Device #" << (deviceIndex + 1) << "/" << devicesCount << std::endl;
            cl_device_id deviceId = devices[deviceIndex];

            size_t deviceNameSize = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_NAME, 0, nullptr, &deviceNameSize));
            std::vector<unsigned char> deviceName(deviceNameSize, 0);
            OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_NAME, deviceNameSize, deviceName.data(), nullptr));
            std::cout << "        Device name: " << deviceName.data() << std::endl;

            cl_device_type deviceType = CL_DEVICE_TYPE_DEFAULT;
            size_t deviceTypeSize = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_TYPE, 0, nullptr, &deviceTypeSize));
            OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_TYPE, deviceTypeSize, &deviceType, nullptr));
            switch (deviceType) {
                case CL_DEVICE_TYPE_CPU:
                    std::cout << "        Device type: " << "Processor" << std::endl;
                    break;
                case CL_DEVICE_TYPE_GPU:
                    std::cout << "        Device type: " << "Graphic card" << std::endl;
                    break;
                default:
                    std::cout << "        Device type: " << "Something strange" << std::endl;
                    break;
            }

            cl_ulong deviceMemSize = 0;
            size_t deviceMemSizeSize = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_GLOBAL_MEM_SIZE, 0, nullptr, &deviceMemSizeSize));
            OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_GLOBAL_MEM_SIZE, deviceMemSizeSize, &deviceMemSize,
                                          nullptr));
            std::cout << "        Device memory size: " << (deviceMemSize >> 20) << " MBs" << std::endl;

            cl_uint clock_frequency = 0;
            size_t clock_frequency_size = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_MAX_CLOCK_FREQUENCY, 0, nullptr, &clock_frequency_size));
            OCL_SAFE_CALL(
                    clGetDeviceInfo(deviceId, CL_DEVICE_MAX_CLOCK_FREQUENCY, clock_frequency_size, &clock_frequency,
                                    nullptr));
            std::cout << "        Device maximum clock frequency: " << clock_frequency << " MHz" << std::endl;

            size_t driverVersionSize = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DRIVER_VERSION, 0, nullptr, &driverVersionSize));
            std::vector<unsigned char> driverVersion(driverVersionSize, 0);
            OCL_SAFE_CALL(
                    clGetDeviceInfo(deviceId, CL_DRIVER_VERSION, driverVersionSize, driverVersion.data(), nullptr));
            std::cout << "        Device driver version: " << driverVersion.data() << std::endl;
        }
    }

    return 0;
}