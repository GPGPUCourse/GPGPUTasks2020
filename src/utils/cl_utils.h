#include <CL/cl.h>
#include <string>

void reportError(cl_int err, const std::string &filename, int line);

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)

std::ostream & operator<<(std::ostream & out, const cl_program program);
cl_device_id getCLDevice(cl_device_type selectedDeviceType = CL_DEVICE_TYPE_GPU);
