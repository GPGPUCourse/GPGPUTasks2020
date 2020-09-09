#include <CL/cl.h>
#include <libclew/ocl_init.h>

#include <vector>
#include <sstream>
#include <iostream>
#include <stdexcept>


constexpr cl_ulong KB = 1024;
constexpr cl_ulong MB = KB * 1024;


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

std::string deviceTypeToString(cl_device_type type) {
    switch (type) {
        case CL_DEVICE_TYPE_CPU: return "CPU";
        case CL_DEVICE_TYPE_GPU: return "GPU";
        case CL_DEVICE_TYPE_ACCELERATOR: return "Accelerator";
        default: return "Default";
    }
}

std::vector<cl_device_id> getDevices(cl_platform_id platform, cl_uint* devicesCount) {
    OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, devicesCount));

    std::vector<cl_device_id> devices(*devicesCount);
    OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, *devicesCount, devices.data(), nullptr));

    return devices;
}


template <typename Character>
std::vector<Character> getDeviceInfoString(cl_device_id device, cl_device_info info) {
    size_t deviceStringSize = 0;
    OCL_SAFE_CALL(clGetDeviceInfo(device, info, 0, nullptr, &deviceStringSize));

    std::vector<Character> deviceString(deviceStringSize, 0);
    OCL_SAFE_CALL(clGetDeviceInfo(device, info, deviceStringSize, deviceString.data(), nullptr));

    return deviceString;
}

template <typename T>
T getDeviceInfoVar(cl_device_id device, cl_device_info info) {
    size_t deviceInfoSize = 0;
    OCL_SAFE_CALL(clGetDeviceInfo(device, info, 0, nullptr, &deviceInfoSize));

    T deviceInfo;
    OCL_SAFE_CALL(clGetDeviceInfo(device, info, deviceInfoSize, &deviceInfo, nullptr));
    return deviceInfo;
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

        /*
        throws CL_INVALID_VALUE -30 error code
        OCL_SAFE_CALL(clGetPlatformInfo(platform, 42, 0, nullptr, &platformNameSize));
         */

        // TODO 1.2
        // Аналогично тому как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:
        std::vector<unsigned char> platformName = getPlatformInfoString<unsigned char>(platform, CL_PLATFORM_NAME);
        std::cout << "    Platform name: " << platformName.data() << std::endl;

        // TODO 1.3
        // Запросите и напечатайте так же в консоль вендора данной платформы
        std::vector<unsigned char> platformVendor = getPlatformInfoString<unsigned char>(platform, CL_PLATFORM_VENDOR);
        std::cout << "    Platform vendor: " << platformVendor.data() << std::endl;

        // TODO 2.1
        // Запросите число доступных устройств данной платформы (аналогично тому как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
        cl_uint devicesCount = 0;
        std::vector<cl_device_id> devices = getDevices(platform, &devicesCount);
        std::cout << "    Number of devices: " << devicesCount << std::endl;

        for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
            std::cout << "    Device #" << (deviceIndex + 1) << "/" << devicesCount << std::endl;
            cl_device_id device = devices[deviceIndex];
            // TODO 2.2
            // Запросите и напечатайте в консоль:
            // - Название устройства
            // - Тип устройства (видеокарта/процессор/что-то странное)
            // - Размер памяти устройства в мегабайтах
            // - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
            std::vector<unsigned char> deviceName = getDeviceInfoString<unsigned char>(device, CL_DEVICE_NAME);
            std::cout << "        Device name: " << deviceName.data() << std::endl;

            auto type = getDeviceInfoVar<cl_device_type>(device, CL_DEVICE_TYPE);
            std::cout << "        Device type: " << deviceTypeToString(type) << std::endl;

            auto localMemSize = getDeviceInfoVar<cl_ulong>(device, CL_DEVICE_LOCAL_MEM_SIZE);
            std::cout << "        Local memory size: " << localMemSize / KB << " KB" << std::endl;

            auto globalMemSize = getDeviceInfoVar<cl_ulong>(device, CL_DEVICE_GLOBAL_MEM_SIZE);
            std::cout << "        Global memory size: " << globalMemSize / MB << " MB" << std::endl;

            std::vector<unsigned char> clVersion = getDeviceInfoString<unsigned char>(device, CL_DEVICE_VERSION);
            std::cout << "        Device OpenCL supported version: " << clVersion.data() << std::endl;

            auto maxGroupSize = getDeviceInfoVar<cl_uint>(device, CL_DEVICE_MAX_WORK_GROUP_SIZE);
            std::cout << "        Maximum number of work-items in a work-group: " << maxGroupSize << std::endl;

            auto maxUnits = getDeviceInfoVar<cl_uint>(device, CL_DEVICE_MAX_COMPUTE_UNITS);
            std::cout << "        Number of parallel compute units: " << maxUnits << std::endl;

            auto maxClockFreq = getDeviceInfoVar<cl_uint>(device, CL_DEVICE_MAX_CLOCK_FREQUENCY);
            std::cout << "        Maximum configured clock frequency: " << maxClockFreq << " MHz" << std::endl;

            auto isCorrSupportPresent = getDeviceInfoVar<cl_bool>(device, CL_DEVICE_ERROR_CORRECTION_SUPPORT);
            std::cout << "        Correction support: " << (isCorrSupportPresent ? "true" : "false") << std::endl;
        }
    }

    return 0;
}