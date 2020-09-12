#include <CL/cl.h>
#include <libclew/ocl_init.h>

#include <vector>
#include <sstream>
#include <iostream>
#include <stdexcept>


template <typename T>
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

template <typename T>
T getDeviceInfo(cl_device_id device, cl_device_info info) {
	size_t deviceInfoSize = 0;
	OCL_SAFE_CALL(clGetDeviceInfo(device, info, 0, nullptr, &deviceInfoSize));

	T deviceInfo;
	OCL_SAFE_CALL(clGetDeviceInfo(device, info, deviceInfoSize, &deviceInfo, nullptr));
	return deviceInfo;
}

std::vector<unsigned char> getDeviceInfo(cl_device_id device, cl_device_info info) {
	size_t deviceInfoSize = 0;
	OCL_SAFE_CALL(clGetDeviceInfo(device, info, 0, nullptr, &deviceInfoSize));

	std::vector<unsigned char> deviceInfo(deviceInfoSize, 0);
	OCL_SAFE_CALL(clGetDeviceInfo(device, info, deviceInfoSize, deviceInfo.data(), nullptr));
	return deviceInfo;
}


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
        // TODO 1.1
        // Попробуйте вместо CL_PLATFORM_NAME передать какое-нибудь случайное число - например 239
        // Т.к. это некорректный идентификатор параметра платформы - то метод вернет код ошибки

		// Для 239 был получен код ошибки -30 

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
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr));
        std::cout << "\tPlatform name: " << platformName.data() << std::endl;

        // TODO 1.3
        // Запросите и напечатайте так же в консоль вендора данной платформы
		size_t vendorNameSize = 0;
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &vendorNameSize));

		std::vector<unsigned char> vendorName(vendorNameSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, vendorNameSize, vendorName.data(), nullptr));
		std::cout << "\tPlatform name: " << vendorName.data() << std::endl;

        // TODO 2.1
        // Запросите число доступных устройств данной платформы (аналогично тому как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
        cl_uint devicesCount = 0;
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));

		std::vector<cl_device_id> devices(devicesCount, 0);
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));

        for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
			std::cout << "\tDevice #" << deviceIndex + 1 << " from " << devicesCount << std::endl;

			cl_device_id deviceId = devices[deviceIndex];

			// TODO 2.2
            // Запросите и напечатайте в консоль:
            // - Название устройства
			std::vector<unsigned char> deviceName = std::move(getDeviceInfo(deviceId, CL_DEVICE_NAME));
			std::cout << "\t\tDevice name: " << deviceName.data() << std::endl;

            // - Тип устройства (видеокарта/процессор/что-то странное)
			size_t deviceTypeSize = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_TYPE, 0, nullptr, &deviceTypeSize));

			cl_device_type deviceType = CL_DEVICE_TYPE_DEFAULT;
			OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_TYPE, deviceTypeSize, &deviceType, nullptr));
			switch (deviceType) {
				case CL_DEVICE_TYPE_CPU:
					std::cout << "\t\tDevice type: CPU" << std::endl;
					break;
				case CL_DEVICE_TYPE_GPU:
					std::cout << "\t\tDevice type: GPU" << std::endl;
					break;
				default:
					std::cout << "\t\tDevice type: something strange was found" << std::endl;
					break;
			}
            
			// - Размер памяти устройства в мегабайтах
			cl_ulong deviceMemSize = getDeviceInfo<cl_ulong>(deviceId, CL_DEVICE_GLOBAL_MEM_SIZE);
			std::cout << "\t\tDevice memory size: " << (deviceMemSize >> 20) << " MBs" << std::endl;
            
			// - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
			// - Max size of memory object allocation in MBs
			cl_ulong deviceeMaxMemAllocSize = getDeviceInfo<cl_ulong>(deviceId, CL_DEVICE_MAX_MEM_ALLOC_SIZE);
			std::cout << "\t\tMax size of memory object allocation: " << (deviceeMaxMemAllocSize >> 20) << " MBs" << std::endl;

			// - Тактовая частота в мегагерцах
			cl_uint clockFreq = getDeviceInfo<cl_ulong>(deviceId, CL_DEVICE_MAX_CLOCK_FREQUENCY);
			std::cout << "\t\tMaximum clock frequency: " << clockFreq << " MHz" << std::endl;

			// - Поддерживаемая версия OpenCL
			std::vector<unsigned char> deviceVersion = std::move(getDeviceInfo(deviceId, CL_DEVICE_VERSION));
			std::cout << "\t\tDevice OpenCL supported version: " << deviceVersion.data() << std::endl;

        }
    }

    return 0;
}
