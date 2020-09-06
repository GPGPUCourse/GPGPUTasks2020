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

template <typename T>
void safeCallDeviceInfo(cl_device_id device, cl_device_info param_name, T* value) {
    size_t size_var = 0;
    OCL_SAFE_CALL(clGetDeviceInfo(device, param_name, 0, nullptr, &size_var));
    OCL_SAFE_CALL(clGetDeviceInfo(device, param_name, size_var, value, nullptr));
}

template <typename T>
void safeCallPlatformInfo(cl_platform_id platform, cl_device_info param_name, T* value) {
    size_t size_var = 0;
    OCL_SAFE_CALL(clGetPlatformInfo(platform, param_name, 0, nullptr, &size_var));
    OCL_SAFE_CALL(clGetPlatformInfo(platform, param_name, size_var, value, nullptr));
}

const size_t MAX_STRING = 100;

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

    // Проверка для пункта 1.1, в цикле делать ее было бы странно
    // Прилетает ошибка CL_INVALID_VALUE так как отправденное в качестве параметра cl_platform_info значение не
    // принадлежит к пулу поддерживаемых запросов.
    try {
        size_t pltNameSize = 0;
        OCL_SAFE_CALL(clGetPlatformInfo(platforms[0], 239, 0, nullptr, &pltNameSize));
    } catch (std::exception& err) {
        std::cerr << err.what() << '\n';
    }

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
        std::vector<unsigned char> platformName(platformNameSize, 0);
        safeCallPlatformInfo(platform, CL_PLATFORM_NAME, platformName.data());
        std::cout << "\tPlatform name: " << platformName.data() << std::endl;


        // TODO 1.3
        // Запросите и напечатайте так же в консоль вендора данной платформы
        std::vector<unsigned char> platformVendor(100, 0);
        safeCallPlatformInfo(platform, CL_PLATFORM_VENDOR, platformVendor.data());
        std::cout << "\tPlatform vendor: " << platformVendor.data() << std::endl;

        // TODO 2.1
        // Запросите число доступных устройств данной платформы (аналогично тому как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
        cl_uint devicesCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
        std::cout << "\tNumber of devices: " << devicesCount << std::endl;
        std::vector<cl_device_id> devices(platformsCount);
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));

        for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
            // TODO 2.2
            // Запросите и напечатайте в консоль:
            // - Название устройства
            // - Тип устройства (видеокарта/процессор/что-то странное)
            // - Размер памяти устройства в мегабайтах
            // - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
            cl_device_id device = devices[deviceIndex];

            std::vector<unsigned char> deviceName(MAX_STRING, 0);
            safeCallDeviceInfo(device, CL_DEVICE_NAME, deviceName.data());
            std::cout << "\tDevice #" << (deviceIndex + 1) << '/' << devicesCount
                << "\n\t\tDevice name: " <<deviceName.data() << std::endl;

            cl_device_type deviceType;
            safeCallDeviceInfo(device, CL_DEVICE_TYPE, &deviceType);
            std::cout << "\t\tDevice type: ";
            if (deviceType == CL_DEVICE_TYPE_CPU) {
                std::cout << "CPU" << std::endl;
            } else if (deviceType == CL_DEVICE_TYPE_GPU) {
                std::cout << "GPU" << std::endl;
            } else if (deviceType == CL_DEVICE_TYPE_ACCELERATOR) {
                std::cout << "Accelerator" << std::endl;
            } else {
                std::cout << "Something else)" << std::endl;
            }

            cl_ulong deviceMem;
            safeCallDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, &deviceMem);
            std::cout << "\t\tDevice memory: " << (deviceMem >> 20) << std::endl;

            std::vector<unsigned char> deviceDriverVersion(MAX_STRING, 0);
            safeCallDeviceInfo(device, CL_DRIVER_VERSION, deviceDriverVersion.data());
            std::cout << "\t\tDevice version: " << deviceDriverVersion.data() << std::endl;

            std::vector<unsigned char> deviceVersion(MAX_STRING, 0);
            safeCallDeviceInfo(device, CL_DEVICE_VERSION, deviceVersion.data());
            std::cout << "\t\tDevice version: " << deviceVersion.data() << std::endl;
        }
    }

    return 0;
}