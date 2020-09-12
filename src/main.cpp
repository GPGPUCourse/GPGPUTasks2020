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
    std::cerr << message << "\n";
    std::cerr << "Error code: " << std::dec << err << "\n";
    throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)


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

    for (size_t platformIndex = 0; platformIndex < platformsCount; ++platformIndex) {
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
        {
          std::vector<unsigned char> platformName(platformNameSize, 0);
          // clGetPlatformInfo(...);
          OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize * sizeof(platformName[10000]), platformName.data(), nullptr));
          std::cout << "    Platform name: " << platformName.data() << std::endl;
        }

        // TODO 1.3
        // Запросите и напечатайте так же в консоль вендора данной платформы
        {
          size_t vendorNameSize = 0;
          OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &vendorNameSize));
          std::vector<char> vendorName(vendorNameSize);
          OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, vendorNameSize * sizeof(vendorName[0]), vendorName.data(), nullptr));
          std::cout << "    Vendor name: " << vendorName.data() << "\n";
        }

        // TODO 2.1
        // Запросите число доступных устройств данной платформы (аналогично тому как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
        cl_uint devicesCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
        std::cout << "    Number of devices: " << devicesCount << "\n";
        std::vector<cl_device_id> devices(devicesCount);
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), 0));

        for (size_t deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
            // TODO 2.2
            // clGetDeviceInfo(devices[deviceIndex], cl_device_info param_name,
            //     size_t param_value_size, void* param_value, size_t * param_value_size_ret);

            // Запросите и напечатайте в консоль:
            // - Название устройства
            {
                size_t deviceNameSize;
                OCL_SAFE_CALL(clGetDeviceInfo(devices[deviceIndex], CL_DEVICE_NAME, 0, nullptr, &deviceNameSize));
                std::vector<char> deviceName(deviceNameSize);
                OCL_SAFE_CALL(clGetDeviceInfo(devices[deviceIndex], CL_DEVICE_NAME, deviceNameSize, deviceName.data(), nullptr));
                std::cout << "    " << deviceName.data() << "\n";
            }

            // - Тип устройства (видеокарта/процессор/что-то странное)
            {
                cl_device_type deviceType;
                OCL_SAFE_CALL(clGetDeviceInfo(devices[deviceIndex], CL_DEVICE_TYPE, sizeof(cl_device_type), &deviceType, nullptr));
                std::cout << "    Device type: ";
                {
                  bool didNotPrinted = true;
                  if (deviceType&CL_DEVICE_TYPE_DEFAULT) {
                    if (didNotPrinted) didNotPrinted = false; else std::cout << " | ";
                    std::cout << "DEFAULT";
                  }
                  if (deviceType&CL_DEVICE_TYPE_CPU) {
                    if (didNotPrinted) didNotPrinted = false; else std::cout << " | ";
                    std::cout << "CPU";
                  }
                  if (deviceType&CL_DEVICE_TYPE_GPU) {
                    if (didNotPrinted) didNotPrinted = false; else std::cout << " | ";
                    std::cout << "GPU";
                  }
                  if (deviceType&CL_DEVICE_TYPE_ACCELERATOR) {
                    if (didNotPrinted) didNotPrinted = false; else std::cout << " | ";
                    std::cout << "ACCELERATOR";
                  }
                  if (didNotPrinted) {
                    std::cout << "UNKNOWN";
                  }
                  std::cout << "\n";
                }
            }

            // - Размер памяти устройства в мегабайтах
            {
              cl_ulong memBytes;
              OCL_SAFE_CALL(clGetDeviceInfo(devices[deviceIndex], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &memBytes, nullptr));
              std::cout << "    Device memory: " << memBytes/1000000.0 << "M" << "\n";
            }

            // - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
            {
              cl_ulong cacheBytes;
              OCL_SAFE_CALL(clGetDeviceInfo(devices[deviceIndex], CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(cl_ulong), &cacheBytes, nullptr));
              std::cout << "    Device cache: " << cacheBytes/1024.0 << "KiB" << "\n";
            }
            {
              cl_ulong maxMemAlloc;
              OCL_SAFE_CALL(clGetDeviceInfo(devices[deviceIndex], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &maxMemAlloc, nullptr));
              std::cout << "    Device max size of memory object allocation: " << maxMemAlloc/1024.0/1024.0 << "MiB" << "\n";
            }
        }
    }

    return 0;
}
