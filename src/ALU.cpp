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
    assert(currentDevice != nullptr);
}

void ALU::initComputation() {

    cl_int *errcode_ret = nullptr;


    const cl_device_id devices []{currentDevice};

    currentContext = clCreateContext(nullptr,
                                     1,
                                     devices,
                                     nullptr,
                                     nullptr,
                                     errcode_ret);

    assert(errcode_ret == nullptr);

    errcode_ret = nullptr;
    currentCommandQueue = clCreateCommandQueue(currentContext,
                                               currentDevice,
                                               0,
                                               errcode_ret);

    assert(errcode_ret == nullptr);
}

void ALU::resetComputation() {

    for (const auto& buffer : usedInBuffers) {
        clReleaseMemObject(buffer);
    }

    for (const auto& buffer : usedOutBuffers) {
        clReleaseMemObject(buffer);
    }

    clReleaseProgram(additiveProgram);

    clReleaseCommandQueue(currentCommandQueue);
    clReleaseContext(currentContext);
}

void ALU::createBuffer(std::vector<float> &data, std::vector<cl_mem>& storage, cl_mem_flags flags) {

    cl_int *error = nullptr;
    auto memBuffer = clCreateBuffer(currentContext,
                                    flags,
                                    sizeof(float) * data.size(),
                                    data.data(),
                                    error);

    assert(memBuffer != nullptr);
    storage.push_back(memBuffer);
}

void ALU::initBuffers(std::vector<float> &as,
                      std::vector<float> &bs) {

    usedOutBuffers.clear();

    createBuffer(as, usedInBuffers, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    createBuffer(bs, usedInBuffers, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);

    usedOutBuffers.clear();
    cl_int *error = nullptr;
    auto outputBuffer = clCreateBuffer(currentContext,
                                       CL_MEM_WRITE_ONLY,
                                       sizeof(float) * as.size(),
                                       nullptr,
                                       error);
    assert(outputBuffer != nullptr);
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
    std::vector<const size_t> length(kernel_sources.length());

    cl_int * error = nullptr;
    additiveProgram = clCreateProgramWithSource(currentContext,
                                                1,
                                                sources.data(),
                                                length.data(),
                                                error);

    assert(error == nullptr);

    cl_device_id devices[] {currentDevice};

    auto buildResult = clBuildProgram(additiveProgram, 1, devices, nullptr, nullptr, nullptr);

    std::cerr << additiveProgram << std::endl;
    assert(buildResult == CL_BUILD_SUCCESS);
}

void ALU::initKernel() {

    cl_int * error = nullptr;
    additiveKernel = clCreateKernel(additiveProgram, "aplusb",  error);
    assert(additiveKernel != nullptr);
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
    clWaitForEvents(1, events);
}
