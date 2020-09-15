#include <CL/cl.h>
#include <libclew/ocl_init.h>

#include <vector>
#include <sstream>
#include <iostream>
#include <stdexcept>


template<typename T>
std::string to_string(T value) {
	std::ostringstream ss;
	ss << value;
	return ss.str();
}

void reportError(cl_int err, const std::string& filename, int line) {
	if (CL_SUCCESS == err)
		return;

	// Таблица с кодами ошибок:
	// libs/clew/CL/cl.h:103
	// P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
	std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
	throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)


void linkOpenCL();

std::unique_ptr<std::vector<cl_platform_id>> getPlatforms();

void showPlatformName(_cl_platform_id *platform);

void showPlatformVendor(_cl_platform_id *platform);

std::unique_ptr<std::vector<cl_device_id>> getDevices(cl_platform_id platform);

void showDeviceName(_cl_device_id *device);

void showOpenCLInfo();

void showPlatform(cl_platform_id const& platform);

void showDevice(cl_device_id const& device);

void showDevices(cl_platform_id const& platform);

void showDeviceType(_cl_device_id *device);

void showDeviceMemorySize(_cl_device_id *device);

void showDeviceNumberOfComputeUnits(_cl_device_id * device);

void showDeviceMaxClockFrequency(_cl_device_id * device);

void showPlatforms(const std::unique_ptr<std::vector<cl_platform_id>>& platforms);

std::string getDeviceType(cl_device_type type);

int main() {
	linkOpenCL();
	showOpenCLInfo();
	return 0;
}

void showOpenCLInfo() {
	auto platforms = getPlatforms();
	std::cout << "Number of OpenCL platforms: " << platforms->size() << std::endl;
	showPlatforms(platforms);
}

void showPlatforms(const std::unique_ptr<std::vector<cl_platform_id>>& platforms) {
	for (auto& platform : *platforms) {
		showPlatform(platform);
	}
}

void showDevices(const cl_platform_id& platform) {
	auto devices = getDevices(platform);
	for (auto& device : *devices) {
		showDevice(device);
	}
}

void showDevice(const cl_device_id& device) {
	std::cout << "\tDevice:" << std::endl;
	showDeviceName(device);
	showDeviceType(device);
	showDeviceMemorySize(device);
	showDeviceNumberOfComputeUnits(device);
	showDeviceMaxClockFrequency(device);
}

void showPlatform(const cl_platform_id& platform) {
	std::cout << "Platform:" << std::endl;
	showPlatformName(platform);
	showPlatformVendor(platform);
	showDevices(platform);
}

void showDeviceName(_cl_device_id *const device) {
	size_t deviceNameSize = 0;
	OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &deviceNameSize));
	std::vector<char> deviceName(deviceNameSize, 0);
	OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, deviceNameSize, deviceName.data(), nullptr));
	std::cout << "        Device name: " << deviceName.data() << std::endl;
}

std::unique_ptr<std::vector<cl_device_id>> getDevices(cl_platform_id platform) {
	cl_uint devicesCount = 0;
	OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
	auto *devices = new std::vector<cl_device_id>(devicesCount);
	OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, (*devices).data(), nullptr));
	return std::unique_ptr<std::vector<cl_device_id>>(devices);
}

void linkOpenCL() {
	if (!ocl_init()) throw std::runtime_error("Can't init OpenCL driver!");
}

void showPlatformName(_cl_platform_id *const platform) {
	size_t platformNameSize = 0;
	OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameSize));
	std::vector<char> platformName(platformNameSize, 0);
	OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr));
	std::cout << "\tName: " << platformName.data() << std::endl;
}

void showPlatformVendor(_cl_platform_id *const platform) {
	size_t platformNameSize = 0;
	OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &platformNameSize));
	std::vector<char> platformName(platformNameSize, 0);
	OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, platformNameSize, platformName.data(), nullptr));
	std::cout << "\tVendor: " << platformName.data() << std::endl;
}

std::unique_ptr<std::vector<cl_platform_id>> getPlatforms() {
	cl_uint platforms_count = 0;
	OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platforms_count));
	auto *platforms = new std::vector<cl_platform_id>(platforms_count);
	OCL_SAFE_CALL(clGetPlatformIDs(platforms_count, (*platforms).data(), nullptr));
	return std::unique_ptr<std::vector<cl_platform_id>>(platforms);
}

void showDeviceType(_cl_device_id *const device) {
	size_t deviceTypeSize = 0;
	OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, 0, nullptr, &deviceTypeSize));
	std::vector<char> deviceName(deviceTypeSize, 0);
	cl_device_type type;
	OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, deviceTypeSize, &type, nullptr));
	std::cout << "\t\tDevice type is " << getDeviceType(type) << std::endl;
}

void showDeviceMemorySize(_cl_device_id *device) {
	size_t deviceTypeSize = 0;
	OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, 0, nullptr, &deviceTypeSize));
	std::vector<char> deviceName(deviceTypeSize, 0);
	cl_ulong cache_size_bytes;
	OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, deviceTypeSize, &cache_size_bytes, nullptr));
	std::cout << "\t\tSize of global memory is " << cache_size_bytes / (1024 * 1024) << " MB" << std::endl;
}

void showDeviceNumberOfComputeUnits(_cl_device_id *const device) {
	size_t deviceTypeSize = 0;
	OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, 0, nullptr, &deviceTypeSize));
	std::vector<char> deviceName(deviceTypeSize, 0);
	cl_uint number_of_compute_units;
	OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, deviceTypeSize, &number_of_compute_units, nullptr));
	std::cout << "\t\tNumber of parallel compute units on the OpenCL device is " << number_of_compute_units << std::endl;
}

void showDeviceMaxClockFrequency(_cl_device_id *const device) {
	size_t deviceTypeSize = 0;
	OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, 0, nullptr, &deviceTypeSize));
	std::vector<char> deviceName(deviceTypeSize, 0);
	cl_uint number_of_compute_units;
	OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, deviceTypeSize, &number_of_compute_units, nullptr));
	std::cout << "\t\tMaximum configured clock frequency of the device is " << number_of_compute_units << " MHz"<< std::endl;
}

std::string getDeviceType(const cl_device_type type) {
	if (type == CL_DEVICE_TYPE_CPU) return "CPU";
	else if (type == CL_DEVICE_TYPE_GPU) return "GPU";
	return "unknown";
}
