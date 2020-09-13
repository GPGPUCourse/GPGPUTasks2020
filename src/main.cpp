#include <CL/cl.h>
#include <libclew/ocl_init.h>

#include <vector>
#include <sstream>
#include <iostream>
#include <stdexcept>


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
    std::cerr << message << std::endl;
    throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)

std::vector<unsigned char> getStringInfo(cl_platform_id platform, cl_platform_info info) {
    size_t size = 0;
    OCL_SAFE_CALL(clGetPlatformInfo(platform, info, 0, nullptr, &size));

    std::vector<unsigned char> result(size, 0);
    OCL_SAFE_CALL(clGetPlatformInfo(platform, info, size, result.data(), nullptr));

    return result;
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

    // Тот же метод используется для того чтобы получить идентификаторы всех платформ - сверьтесь с документацией, что это сделано верно:
    std::vector<cl_platform_id> platforms(platformsCount);
    OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));

    for (int platformIndex = 0; platformIndex < platformsCount; ++platformIndex) {
        std::cout << "Platform #" << (platformIndex + 1) << "/" << platformsCount << std::endl;
        cl_platform_id platform = platforms[platformIndex];

        std::vector<unsigned char> platformName = getStringInfo(platform, CL_PLATFORM_NAME);
        std::cout << "    Platform name: " << platformName.data() << std::endl;

        std::vector<unsigned char> platformVendor = getStringInfo(platform, CL_PLATFORM_VENDOR);
        std::cout << "    Platform vendor: " << platformVendor.data() << std::endl;

        cl_uint devicesCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));

        std::vector<cl_device_id> devices(devicesCount);
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), &devicesCount));

        for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
            std::cout << "        Device #" << (deviceIndex + 1) << "/" << devicesCount << std::endl;

            cl_device_id device = devices[deviceIndex];

            size_t nameLength = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &nameLength));
            std::vector<unsigned char> name(nameLength, 0);
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, nameLength, name.data(), nullptr));
            std::cout << "         Name: " << name.data() << std::endl;

            cl_device_type deviceType = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &deviceType, nullptr));
            std::string deviceTypeRepr = "";
            switch (deviceType) {
                case CL_DEVICE_TYPE_CPU:
                    deviceTypeRepr = "CPU";
                    break;
                case CL_DEVICE_TYPE_ACCELERATOR:
                    deviceTypeRepr = "ACCELERATOR";
                    break;
                case CL_DEVICE_TYPE_GPU:
                    deviceTypeRepr = "GPU";
                    break;
                default:
                    deviceTypeRepr = "UNKNOWN";
            }
            std::cout << "         Type: " << deviceTypeRepr << std::endl;

            cl_ulong memSize = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(memSize), &memSize, nullptr));
            std::cout << "         Memory (MB): " << (memSize / 1024 / 1024) << std::endl;

            cl_bool available = false;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_AVAILABLE, sizeof(cl_bool), &available, nullptr));
            std::cout << "         Is " << (!available ? "NOT " : "") << "available." << std::endl;

            size_t driverVersionLen = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DRIVER_VERSION, 0, nullptr, &driverVersionLen));
            std::vector<unsigned char> driverVersion(driverVersionLen, 0);
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DRIVER_VERSION, driverVersionLen, driverVersion.data(), nullptr));
            std::cout << "         Driver version: " << driverVersion.data() << std::endl;
        }
    }

    return 0;
}