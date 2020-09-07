#include <CL/cl.hpp>
#include <libclew/ocl_init.h>

#include <functional>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

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

template<cl_device_info info, typename ProcessFunc>
void printDeviceInfo(const cl::Device &device, std::string description, const ProcessFunc &processFunc) {
    cl_int result;
    const auto resultValue = device.getInfo<info>(&result);
    OCL_SAFE_CALL(result);
    std::cout << "        " << std::move(description) << ": " << processFunc(resultValue) << std::endl;
}

template<cl_device_info info>
void printDeviceInfo(const cl::Device &device, std::string description) {
    using T = decltype(device.getInfo<info>(nullptr));
    printDeviceInfo<info>(device, description, [](T value) { return value; });
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
    std::vector<cl::Platform> platforms;
    OCL_SAFE_CALL(cl::Platform::get(&platforms));

    std::cout << "Number of OpenCL platforms: " << platforms.size() << std::endl;

    // Тот же метод используется для того чтобы получить идентификаторы всех платформ - сверьтесь с документацией, что это сделано верно:
    for (size_t platformIndex = 0; platformIndex < platforms.size(); ++platformIndex) {
        std::cout << "Platform #" << (platformIndex + 1) << "/" << platforms.size() << ":" << std::endl;
        const auto &platform = platforms.at(platformIndex);
        
        const auto printPlatformInfo = [&platform](std::string description, cl_platform_info info) {
            static std::string platformInfoString;
            OCL_SAFE_CALL(platform.getInfo(info, &platformInfoString));
            std::cout << "    " << description << ": " << platformInfoString << std::endl;
        };
        
        // Откройте документацию по "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformInfo"
        // Не забывайте проверять коды ошибок с помощью макроса OCL_SAFE_CALL
        // size_t platformNameSize = 0;
        // OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameSize));
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
        printPlatformInfo("Name", CL_PLATFORM_NAME);

        // TODO 1.3
        // Запросите и напечатайте так же в консоль вендора данной платформы
        printPlatformInfo("Vendor", CL_PLATFORM_VENDOR);

        // TODO 2.1
        // Запросите число доступных устройств данной платформы (аналогично тому как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
        std::vector<cl::Device> devices;
        OCL_SAFE_CALL(platform.getDevices(CL_DEVICE_TYPE_ALL, &devices));
        
        for (size_t deviceIndex = 0; deviceIndex < devices.size(); ++deviceIndex) {
            std::cout << "    Device #" << (deviceIndex + 1) << "/" << devices.size() << ":" << std::endl;
            const auto &device = devices.at(deviceIndex);
        
            // const auto printDeviceInfo = [&device](std::string description, cl_device_info info) {
            //     // static std::string deviceInfoString;
            //     // OCL_SAFE_CALL(device.getInfo(info, &deviceInfoString));
            //     cl_int result;
            //     const auto resultValue = device.getInfo<info>(&result);
            //     OCL_SAFE_CALL(result);
            //     std::cout << "        Device " << description << ": " << resultValue << std::endl;
            // };
    
            // TODO 2.2
            // Запросите и напечатайте в консоль:
            // - Название устройства
            // - Тип устройства (видеокарта/процессор/что-то странное)
            // - Размер памяти устройства в мегабайтах
            // - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
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

    return 0;
}