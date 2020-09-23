#include <CL/cl.h>
#include <libclew/ocl_init.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>

#include <vector>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <cassert>
#include <tuple>


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

template<class T = char, class P = size_t, class... Ps, class... Rs>
std::pair<std::vector<T>, P> extractFromFunc(cl_int (*clFunc)(Ps...), Rs... args)
{
    P paramName = 0;
    OCL_SAFE_CALL(clFunc(args..., 0, nullptr, &paramName));
    std::vector<T> returnVec(paramName, (T) 0);
    OCL_SAFE_CALL(clFunc(args..., paramName, returnVec.data(), nullptr));
    return std::make_pair(returnVec, paramName);
}

std::string getDeviceTypeName(cl_device_type type) {
    switch (type) {
        case CL_DEVICE_TYPE_DEFAULT:
            return "CL_DEVICE_TYPE_DEFAULT";
        case CL_DEVICE_TYPE_CPU:
            return "CL_DEVICE_TYPE_CPU";
        case CL_DEVICE_TYPE_GPU:
            return "CL_DEVICE_TYPE_GPU";
        case CL_DEVICE_TYPE_ACCELERATOR:
            return "CL_DEVICE_TYPE_ACCELERATOR";
        default:
            return "CL_DEVICE_TYPE_ALL";
    }
}

int main()
{
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    cl_platform_id platform = extractFromFunc<cl_platform_id, cl_uint>(&clGetPlatformIDs).first[0];

    std::vector<cl_device_id> devices;
    cl_uint devicesCount = 0;
    cl_device_type deviceType = CL_DEVICE_TYPE_GPU;
    cl_int errcode;

    errcode = clGetDeviceIDs(platform, deviceType, 0, nullptr, &devicesCount);
    if (errcode == CL_DEVICE_NOT_FOUND) {
        deviceType = CL_DEVICE_TYPE_CPU;
        errcode = clGetDeviceIDs(platform, deviceType, 0, nullptr, &devicesCount);
    }
    OCL_SAFE_CALL(errcode);

    cl_device_id device = extractFromFunc<cl_device_id, cl_uint>(&clGetDeviceIDs, platform, deviceType).first[0];

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &errcode);
    OCL_SAFE_CALL(errcode);

    cl_command_queue command_queue = clCreateCommandQueue(context, device, 0, &errcode);
    OCL_SAFE_CALL(errcode);

    unsigned int n = 100 * 1000 * 1000;
    std::vector<float> as(n, 0);
    std::vector<float> bs(n, 0);
    std::vector<float> cs(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.nextf();
        bs[i] = r.nextf();
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    size_t mem_size = sizeof(float) * n;

    cl_mem as_mem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mem_size, as.data(), &errcode);
    OCL_SAFE_CALL(errcode);
    cl_mem bs_mem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mem_size, bs.data(), &errcode);
    OCL_SAFE_CALL(errcode);
    cl_mem cs_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, mem_size, cs.data(), &errcode);
    OCL_SAFE_CALL(errcode);

    std::string kernel_sources;
    std::ifstream file("src/cl/aplusb.cl");
    kernel_sources = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    if (kernel_sources.size() == 0) {
        throw std::runtime_error("Empty source file! May be you forgot to configure working directory properly?");
    }

    const char *kernel_sources_c_str = kernel_sources.c_str();
    cl_program program = clCreateProgramWithSource(context, 1, &kernel_sources_c_str, nullptr, &errcode);
    OCL_SAFE_CALL(errcode);

    errcode = clBuildProgram(program, 1, &device, "", nullptr, nullptr);
    size_t logSize;
    std::vector<char> log;
    tie(log, logSize) = extractFromFunc(&clGetProgramBuildInfo, program, device, CL_PROGRAM_BUILD_LOG);
    if (logSize > 1) {
        std::cout << "--------------------- Log ---------------------" << std::endl;
        std::cout << log.data() << std::endl;
        std::cout << "-----------------------------------------------" << std::endl;
    }
    OCL_SAFE_CALL(errcode);

    cl_kernel kernel = clCreateKernel(program, "aplusb", &errcode);
    OCL_SAFE_CALL(errcode);

    OCL_SAFE_CALL(clSetKernelArg(kernel, 0, sizeof(cl_mem), &as_mem));
    OCL_SAFE_CALL(clSetKernelArg(kernel, 1, sizeof(cl_mem), &bs_mem));
    OCL_SAFE_CALL(clSetKernelArg(kernel, 2, sizeof(cl_mem), &cs_mem));
    OCL_SAFE_CALL(clSetKernelArg(kernel, 3, sizeof(unsigned int), &n));

    {
        size_t workGroupSize = 128;
        size_t global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        timer t;
        for (unsigned int i = 0; i < 20; ++i) {
            cl_event event;
            OCL_SAFE_CALL(clEnqueueNDRangeKernel(command_queue, kernel, 1, nullptr, &global_work_size, &workGroupSize, 0, nullptr, &event));
            OCL_SAFE_CALL(clWaitForEvents(1, &event));
            t.nextLap();
            OCL_SAFE_CALL(clReleaseEvent(event));
        }

        std::cout << "Kernel average time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GFlops: " << n / (t.lapAvg() * 1000 * 1000 * 1000) << std::endl;
        std::cout << "VRAM bandwidth: " << 3.0 * mem_size / (t.lapAvg() * 1024 * 1024 * 1024) << " GB/s" << std::endl;
    }

    {
        timer t;
        for (unsigned int i = 0; i < 20; ++i) {
            OCL_SAFE_CALL(clEnqueueReadBuffer(command_queue, cs_mem, true, 0, mem_size, cs.data(), 0, nullptr, nullptr));
            t.nextLap();
        }
        std::cout << "Result data transfer time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "VRAM -> RAM bandwidth: " << mem_size / (t.lapAvg() * 1024 * 1024 * 1024) << " GB/s" << std::endl;
    }

    for (unsigned int i = 0; i < n; ++i) {
        if (cs[i] != as[i] + bs[i]) {
            throw std::runtime_error("CPU and GPU results differ!");
        }
    }

    OCL_SAFE_CALL(clReleaseKernel(kernel));
    OCL_SAFE_CALL(clReleaseProgram(program));
    OCL_SAFE_CALL(clReleaseMemObject(cs_mem));
    OCL_SAFE_CALL(clReleaseMemObject(bs_mem));
    OCL_SAFE_CALL(clReleaseMemObject(as_mem));
    OCL_SAFE_CALL(clReleaseCommandQueue(command_queue));
    OCL_SAFE_CALL(clReleaseContext(context));

    return 0;
}
