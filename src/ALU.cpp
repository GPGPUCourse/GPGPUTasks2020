#include "ALU.h"
#include <CL/cl.h>
#include <libutils/fast_random.h>

#include <vector>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <cassert>

#include "utils/cl_utils.h"

void ALU::selectDevice() {
    currentDevice = getCLDevice();
    if(currentDevice == nullptr) {
        currentDevice = getCLDevice(CL_DEVICE_TYPE_CPU);
    }

    if(currentDevice == nullptr) {
        throw std::runtime_error("No proper device found");
    }
}

void ALU::initComputation() {

    cl_int errcode_ret = CL_INVALID_VALUE;
    const cl_device_id devices []{currentDevice};

    currentContext = clCreateContext(nullptr,
                                     1,
                                     devices,
                                     nullptr,
                                     nullptr,
                                     &errcode_ret);

    if(errcode_ret != CL_SUCCESS) {
        throw std::runtime_error("Cannot create device context");
    }

    errcode_ret = CL_INVALID_VALUE;
    currentCommandQueue = clCreateCommandQueue(currentContext,
                                               currentDevice,
                                               0,
                                               &errcode_ret);

    if(errcode_ret != CL_SUCCESS) {
        throw std::runtime_error("Cannot create device command queue");
    }
}

void ALU::resetComputation() {

    resetBuffers();

    OCL_SAFE_CALL(clReleaseProgram(additiveProgram));
    OCL_SAFE_CALL(clReleaseKernel(additiveKernel));
    OCL_SAFE_CALL(clReleaseCommandQueue(currentCommandQueue));
    OCL_SAFE_CALL(clReleaseContext(currentContext));
}

void ALU::resetBuffers() {

    for (const auto& buffer : usedInBuffers) {
        OCL_SAFE_CALL(clReleaseMemObject(buffer));
    }
    usedInBuffers.clear();

    for (const auto& buffer : usedOutBuffers) {
        OCL_SAFE_CALL(clReleaseMemObject(buffer));
    }
    usedOutBuffers.clear();
}


void ALU::createReadBuffer(std::vector<float> &data, std::vector<cl_mem>& storage, cl_mem_flags flags) {

    cl_int error = CL_INVALID_VALUE;
    auto memBuffer = clCreateBuffer(currentContext,
                                    flags,
                                    sizeof(float) * data.size(),
                                    nullptr,
                                    &error);

    if(memBuffer == nullptr) {
        throw std::runtime_error("Cannot allocate buffer");
    }

    storage.push_back(memBuffer);

    cl_event event;

    OCL_SAFE_CALL(clEnqueueWriteBuffer(currentCommandQueue,
                                       memBuffer,
                                       CL_FALSE,
                                       0,
                                       data.size() * sizeof(float),
                                       data.data(),
                                       0,
                                       nullptr,
                                       &event));

    cl_event events[]{event};
    OCL_SAFE_CALL(clWaitForEvents(1, events));
    OCL_SAFE_CALL(clReleaseEvent(event));
}

void ALU::initBuffers(std::vector<float> &as,
                      std::vector<float> &bs) {

    resetBuffers();

    createReadBuffer(as, usedInBuffers, CL_MEM_READ_ONLY);
    createReadBuffer(bs, usedInBuffers, CL_MEM_READ_ONLY);

    cl_int error = CL_INVALID_VALUE;
    auto outputBuffer = clCreateBuffer(currentContext,
                                       CL_MEM_WRITE_ONLY,
                                       sizeof(float) * as.size(),
                                       nullptr,
                                       &error);
    if(outputBuffer == nullptr) {
        throw std::runtime_error("Cannot allocate buffer");
    }
    usedOutBuffers.push_back(outputBuffer);
}

void ALU::initProgram() {

    std::string kernel_sources;
    {
        std::ifstream file("src/cl/aplusb.cl");
        kernel_sources = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        if (kernel_sources.size() == 0) {
            throw std::runtime_error("Empty source file! May be you forgot to configure working directory properly?");
        }
        std::cout << kernel_sources << std::endl;
    }

    std::vector<const char *> sources { kernel_sources.c_str() };
    std::vector<size_t> length(kernel_sources.length());

    cl_int error = CL_INVALID_VALUE;
    additiveProgram = clCreateProgramWithSource(currentContext,
                                                1,
                                                sources.data(),
                                                length.data(),
                                                &error);

    if(error != CL_SUCCESS) {
        throw std::runtime_error("Cannot create program");
    }

    cl_device_id devices[] {currentDevice};

    auto buildResult = clBuildProgram(additiveProgram, 1, devices, nullptr, nullptr, nullptr);

    std::cerr << additiveProgram << std::endl;
    if(buildResult != CL_BUILD_SUCCESS) {
        throw std::runtime_error("Cannot build program");
    }
}

void ALU::initKernel() {

    cl_int error = CL_INVALID_VALUE;
    additiveKernel = clCreateKernel(additiveProgram, "aplusb",  &error);
    if(error != CL_SUCCESS) {
        throw std::runtime_error("Cannot create kernel");
    }
}

void ALU::setAddOperationBuffers(std::vector<float> &as,
                                 std::vector<float> &bs) {

    initBuffers(as, bs);

    clSetKernelArg(additiveKernel,
                   0,
                   sizeof(cl_mem),
                   &usedInBuffers[0]);

    clSetKernelArg(additiveKernel,
                   1,
                   sizeof(cl_mem),
                   &usedInBuffers[1]);

    clSetKernelArg(additiveKernel,
                   2,
                   sizeof(cl_mem),
                   &usedOutBuffers[0]);

    unsigned int dim = as.size();
    OCL_SAFE_CALL(clSetKernelArg(additiveKernel,
                                 3,
                                 sizeof(unsigned int),
                                 &dim));
}

void ALU::add(size_t globalWorkSize, size_t workGroupSize) {

    cl_event event;

    int err = clEnqueueNDRangeKernel(currentCommandQueue,
                                     additiveKernel,
                                     1,
                                     nullptr,
                                     &globalWorkSize,
                                     &workGroupSize,
                                     0,
                                     nullptr,
                                     &event);

    if(err != CL_SUCCESS) {
        throw std::runtime_error("Failed to execute kernel");
    }
    cl_event events[]{event};

    OCL_SAFE_CALL(clWaitForEvents(1, events));
    OCL_SAFE_CALL(clReleaseEvent(event));
}

void ALU::readResult(std::vector<float> &cs) {

    cl_event event;
    OCL_SAFE_CALL(clEnqueueReadBuffer(currentCommandQueue,
                                      usedOutBuffers[0],
                                      CL_FALSE,
                                      0,
                                      cs.size() * sizeof(float),
                                      cs.data(),
                                      0,
                                      nullptr,
                                      &event));

    cl_event events[]{event};
    OCL_SAFE_CALL(clWaitForEvents(1, events));
    OCL_SAFE_CALL(clReleaseEvent(event));
}
