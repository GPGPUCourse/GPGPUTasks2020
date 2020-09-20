#include <CL/cl.h>
#include <libclew/ocl_init.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>

#include <vector>
#include <sstream>
#include <iostream>
#include <stdexcept>

#include "utils/cl_utils.h"
#include "ALU.h"

int main() {

    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    unsigned int n = 1000*1000*100;

    std::vector<float> as(n, 0);
    std::vector<float> bs(n, 0);
    std::vector<float> cs(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.nextf();
        bs[i] = r.nextf();
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    ALU alu{};

    alu.setAddOperationBuffers(as, bs);

    size_t workGroupSize = 128;

    size_t global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
    timer t;
    for (unsigned int i = 0; i < 20; ++i) {
        alu.add(global_work_size, workGroupSize);
        t.nextLap();
    }
    std::cout << "Kernel average time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
    std::cout << std::endl;

    auto gflops = (n / t.lapAvg()) * pow(10, -9);

    std::cout << "GFlops: " << gflops << std::endl;

    auto kernelVRAMBandwidth = (3*n*sizeof(float) / pow(1024, 3)) / t.lapAvg();
    std::cout << "VRAM bandwidth: " << kernelVRAMBandwidth << " GB/s" << std::endl;

    timer readTimer;
    for (unsigned int i = 0; i < 20; ++i) {
        alu.readResult(cs);
        readTimer.nextLap();
    }

    auto readVRAMBandwidth = (cs.size()*sizeof(float) / pow(1024, 3)) / readTimer.lapAvg();

    std::cout << "Result data transfer time: " << readTimer.lapAvg() << "+-" << readTimer.lapStd() << " s" << std::endl;
    std::cout << "VRAM -> RAM bandwidth: " << readVRAMBandwidth << " GB/s" << std::endl;

    for (unsigned int i = 0; i < n; ++i) {
        if (cs[i] != as[i] + bs[i]) {
            throw std::runtime_error("CPU and GPU results differ!");
        }
    }
}