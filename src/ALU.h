#include <CL/cl.h>
#include <vector>

class ALU {

public:
    ALU() {
        selectDevice();
        initComputation();
        initProgram();
        initKernel();
    }

    ~ALU() {
        resetComputation();
    }

    void setAddOperationBuffers(std::vector<float>& as,
                                std::vector<float>& bs);

    void add(size_t globalWorkSize, size_t workGroupSize);
    void readResult(std::vector<float>& cs);

private:

    cl_device_id currentDevice;
    cl_context currentContext;
    cl_command_queue currentCommandQueue;
    cl_program additiveProgram;
    cl_kernel additiveKernel;

    std::vector<cl_mem> usedInBuffers;
    std::vector<cl_mem> usedOutBuffers;

private:

    void selectDevice();
    void initComputation();
    void initBuffers(std::vector<float>& as,
                     std::vector<float>& bs);
    void createBuffer(std::vector<float>& data,
                      std::vector<cl_mem>& storage,
                      cl_mem_flags flags);

    void initProgram();
    void initKernel();

    void resetComputation();
};