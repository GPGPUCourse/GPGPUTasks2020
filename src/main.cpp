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
    throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)

std::string getDeviceStringParam(cl_device_id deviceId, cl_device_info paramName) {
    size_t paramValueSize = 0;
    OCL_SAFE_CALL(clGetDeviceInfo(deviceId, paramName, 0, nullptr, &paramValueSize));
    std::vector<unsigned char> paramValue(paramValueSize, 0);
    OCL_SAFE_CALL(clGetDeviceInfo(deviceId, paramName, paramValueSize, paramValue.data(), nullptr));
    return std::string(paramValue.begin(), paramValue.end());
}

std::string getPlatformStringParam(cl_platform_id platformId, cl_platform_info paramName) {
    size_t paramValueSize = 0;
    OCL_SAFE_CALL(clGetPlatformInfo(platformId, paramName, 0, nullptr, &paramValueSize));
    std::vector<unsigned char> paramValue(paramValueSize, 0);
    OCL_SAFE_CALL(clGetPlatformInfo(platformId, paramName, paramValueSize, paramValue.data(), nullptr));
    return std::string(paramValue.begin(), paramValue.end());
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

        // Откройте документацию по "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformInfo"
        // Не забывайте проверять коды ошибок с помощью макроса OCL_SAFE_CALL
        size_t platformNameSize = 0;
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameSize));
        // TODO 1.1
        // Попробуйте вместо CL_PLATFORM_NAME передать какое-нибудь случайное число - например 239
        // Т.к. это некорректный идентификатор параметра платформы - то метод вернет код ошибки
        // Макрос OCL_SAFE_CALL заметит это, и кинет ошибку с кодом
        // Откройте таблицу с кодами ошибок:
        // libs/clew/CL/cl.h:103
        // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
        // Найдите там нужный код ошибки и ее название
        // Затем откройте документацию по clGetPlatformInfo и в секции Errors найдите ошибку, с которой столкнулись
        // в документации подробно объясняется, какой ситуации соответствует данная ошибка, и это позволит проверив код понять чем же вызвана данная ошибка (не корректным аргументом param_name)
        // Обратите внимание что в этом же libs/clew/CL/cl.h файле указаны всевоможные defines такие как CL_DEVICE_TYPE_GPU и т.п.

        // TODO 1.2
        // Аналогично тому как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:
        std::cout << "    Platform name: " << getPlatformStringParam(platform, CL_PLATFORM_NAME) << std::endl;

        // TODO 1.3
        // Запросите и напечатайте так же в консоль вендора данной платформы
        std::cout << "    Platform vendor: " << getPlatformStringParam(platform, CL_PLATFORM_VENDOR) << std::endl;

        // TODO 2.1
        // Запросите число доступных устройств данной платформы (аналогично тому как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
        cl_uint devicesCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
        std::cout << "    Number of available devices: " << devicesCount << std::endl;
        std::vector<cl_device_id> devices(devicesCount);
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));

        for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
            // TODO 2.2
            // Запросите и напечатайте в консоль:
            // - Название устройства
            std::cout << "        Device name: " << getDeviceStringParam(devices[deviceIndex], CL_DEVICE_NAME)<< std::endl;
            // - Тип устройства (видеокарта/процессор/что-то странное)
            cl_device_type deviceType = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(devices[deviceIndex], CL_DEVICE_TYPE, sizeof(cl_device_type), &deviceType,
                                          nullptr));
            std::cout << "        Device type: ";
            std::vector<std::string> type_strs = {"DEFAULT", "CPU", "GPU", "ACCELERATOR"};
            for(uint off = 0; off != 4; ++off) {
                if ((deviceType >> off) % 2) {
                    std::cout << type_strs[off] << " ";
                }
            }
            std::cout << std::endl;

            // - Размер памяти устройства в мегабайтах
            cl_ulong deviceMem = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(devices[deviceIndex], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &deviceMem,
                                          nullptr));
            std::cout << "        Device memory: " << double(deviceMem) / (1u << 20u) << "MiB" << std::endl;

            // - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
            std::cout << "        Device profile: " << getDeviceStringParam(devices[deviceIndex], CL_DEVICE_PROFILE)<< std::endl;

            cl_bool deviceAvailable = false;
            OCL_SAFE_CALL(clGetDeviceInfo(devices[deviceIndex], CL_DEVICE_AVAILABLE, sizeof(cl_bool), &deviceAvailable,
                                          nullptr));
            std::cout << "        Device available: " << deviceAvailable << std::endl;

            cl_ulong deviceLocalMem = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(devices[deviceIndex], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &deviceLocalMem,
                                          nullptr));
            std::cout << "        Device local memory: " << deviceLocalMem << "B" << std::endl;

            cl_ulong deviceMaxComputeUnits = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(devices[deviceIndex], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_ulong), &deviceMaxComputeUnits,
                                          nullptr));
            std::cout << "        Device max compute units: " << deviceMaxComputeUnits << std::endl;

        }
    }

    return 0;
}