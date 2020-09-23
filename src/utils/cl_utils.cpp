#include "cl_utils.h"
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

cl_device_id getCLDevice(cl_device_type selectedDeviceType) {
    cl_uint platformsCount = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
    std::cout << "Number of OpenCL platforms: " << platformsCount << std::endl;

    std::vector<cl_platform_id> platforms {platformsCount};
    OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));

    cl_device_id selectedDevice = nullptr;

    auto platformsIterator = platforms.begin();
    cl_uint currentMaxComputeUnits = 0;
    while(platformsIterator != platforms.end()) {

        auto platform = *platformsIterator;

        std::vector<unsigned char> platformName = cl_entity_info(static_cast<cl_platform_id>(platform), CL_PLATFORM_NAME, clGetPlatformInfo);
        std::cout << "\tPlatform name: " << platformName.data() << std::endl;

        std::vector<unsigned char> vendorName = cl_entity_info(platform, CL_PLATFORM_VENDOR, clGetPlatformInfo);
        std::cout << "\tVendor name: " << vendorName.data() << std::endl;

        cl_uint devicesCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));

        std::vector<cl_device_id> devices {devicesCount};
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));

        auto devicesIterator = devices.begin();
        while(devicesIterator != devices.end()) {

            cl_device_id device = *devicesIterator;

            auto deviceName = cl_entity_info(device, CL_DEVICE_NAME, clGetDeviceInfo);
            std::cout << "\t\tDevice name: " << deviceName.data() << std::endl;
            auto deviceType = cl_entity_info<cl_device_type>(device, CL_DEVICE_TYPE, clGetDeviceInfo);
            std::cout << "\t\tDevice type: ";

            auto maxComputeUnits = cl_entity_info<cl_uint>(device, CL_DEVICE_MAX_COMPUTE_UNITS, clGetDeviceInfo);

            if (selectedDeviceType == deviceType && currentMaxComputeUnits < maxComputeUnits) {
                selectedDevice = device;
            }
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

            std::cout << "\t\tDevice max compute units: " << round(maxComputeUnits) << std::endl;

            std::cout << "—————————————" << std::endl;

            devicesIterator++;
        }
        platformsIterator++;
    }
    return selectedDevice;
}

std::ostream & operator<<(std::ostream & out, const cl_program program) {

    cl_uint numDevices = 0;
    OCL_SAFE_CALL(clGetProgramInfo(program,
                                   CL_PROGRAM_NUM_DEVICES,
                                   sizeof(numDevices),
                                   &numDevices,
                                   nullptr));

    std::vector<cl_device_id> devices{numDevices};

    OCL_SAFE_CALL(clGetProgramInfo(program,
                                   CL_PROGRAM_DEVICES,
                                   sizeof(cl_device_id) * devices.size(),
                                   devices.data(),
                                   nullptr));

    size_t logSize;

    OCL_SAFE_CALL(clGetProgramBuildInfo(program,
                                        devices[0],
                                        CL_PROGRAM_BUILD_LOG,
                                        0,
                                        nullptr,
                                        &logSize));

    std::vector<char> log(logSize);
    OCL_SAFE_CALL(clGetProgramBuildInfo(program,
                                        devices[0],
                                        CL_PROGRAM_BUILD_LOG,
                                        logSize,
                                        log.data(),
                                        nullptr));

    if (logSize > 1) {
        out << "Program build log:" << std::endl;
        out << log.data() << std::endl;
    }

    return out;
}