#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>

#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/max_prefix_sum_cl.h"

#include <cassert>

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

#define VALS_IN_STEP 16
#define WORKGROUP_SIZE 128

struct MyCPUprefResult {
    int max_sum;
    unsigned int result;
};

MyCPUprefResult myCPUpref(const std::vector<int>& as) {
    unsigned int current_n = as.size();
    unsigned int next_n = gpu::divup(current_n, VALS_IN_STEP);
    std::vector<unsigned int> inp_maxpref_i(current_n);
    for (unsigned i = 0; i < current_n; ++i) inp_maxpref_i[i] = i + 1;
    std::vector<int> inp_maxpref_v(as);
    std::vector<int> inp_sum(as);
    std::vector<unsigned int> out_maxpref_i(next_n);
    std::vector<int> out_maxpref_v(next_n);
    std::vector<int> out_sum(next_n);

    while (current_n > 1) {
        for (unsigned int loc_i = 0; loc_i < next_n; ++loc_i) {
            unsigned int loc_start = loc_i * VALS_IN_STEP;
            int maxpref_v = std::numeric_limits<int>::min();
            unsigned int maxpref_i = -1;
            int sum = 0;
            int sum_prev = 0;
            for (int i = loc_start; i < loc_start + VALS_IN_STEP && i < current_n; ++i) {
                sum = inp_maxpref_v[i] + sum_prev;
                sum_prev += inp_sum[i];
                if (sum > maxpref_v) {
                    maxpref_v = sum;
                    maxpref_i = inp_maxpref_i[i];
                }
            }
            out_maxpref_i[loc_i] = maxpref_i;
            out_maxpref_v[loc_i] = maxpref_v;
            out_sum[loc_i] = sum_prev;
        }

        current_n = next_n;
        next_n = gpu::divup(current_n, VALS_IN_STEP);

        std::swap(inp_maxpref_i, out_maxpref_i);
        std::swap(inp_maxpref_v, out_maxpref_v);
        std::swap(inp_sum, out_sum);
    }

    int max_sum = inp_maxpref_v[0];
    unsigned int result = inp_maxpref_i[0];

    return {max_sum, result};
}

int main(int argc, char **argv)
{
    int benchmarkingIters = 10;
    int max_n = (1 << 24);

    for (int n = 2; n <= max_n; n *= 2) {
        std::cout << "______________________________________________" << std::endl;
        int values_range = std::min(1023, std::numeric_limits<int>::max() / n);
        std::cout << "n=" << n << " values in range: [" << (-values_range) << "; " << values_range << "]" << std::endl;

        std::vector<int> as(n, 0);
        FastRandom r(n);
        for (int i = 0; i < n; ++i) {
            as[i] = r.next(-values_range, values_range);
        }

        int reference_max_sum;
        unsigned int reference_result;
        {
            int max_sum = std::numeric_limits<int>::min();
            int sum = 0;
            unsigned int result = 0;
            for (unsigned int i = 0; i < n; ++i) {
                sum += as[i];
                if (sum > max_sum) {
                    max_sum = sum;
                    result = i + 1;
                }
            }
            reference_max_sum = max_sum;
            reference_result = result;
        }
        std::cout << "Max prefix sum: " << reference_max_sum << " on prefix [0; " << reference_result << ")" << std::endl;

        {
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                int max_sum = std::numeric_limits<int>::min();
                int sum = 0;
                unsigned int result = 0;
                for (unsigned int i = 0; i < n; ++i) {
                    sum += as[i];
                    if (sum > max_sum) {
                        max_sum = sum;
                        result = i + 1;
                    }
                }
                EXPECT_THE_SAME(reference_max_sum, max_sum, "CPU result should be consistent!");
                EXPECT_THE_SAME(reference_result, result, "CPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "CPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            // TODO: implement on OpenCL
            unsigned int current_n = n;
            unsigned int next_n = gpu::divup(current_n, VALS_IN_STEP);

            gpu::Device device = gpu::chooseGPUDevice(argc, argv);
            gpu::Context context;
            context.init(device.device_id_opencl);
            context.activate();

            std::string defines = "-D VALS_IN_STEP=" + std::to_string(VALS_IN_STEP)
                                + " -D WORKGROUP_SIZE=" + std::to_string(WORKGROUP_SIZE)
                                + " -D INT_MIN=" + std::to_string(std::numeric_limits<int>::min());
            ocl::Kernel kernel(max_prefix_sum_kernel,
                                max_prefix_sum_kernel_length,
                                "max_prefix_sum", defines);

            bool printLog = false;
            kernel.compile(printLog);

            gpu::gpu_mem_32u inp_maxpref_i
                            ,out_maxpref_i;
            gpu::gpu_mem_32i inp_maxpref_v
                            ,inp_sum
                            ,out_maxpref_v
                            ,out_sum;
            gpu::gpu_mem_32u const_maxpref_i;
            gpu::gpu_mem_32i const_maxpref_v
                            ,const_sum;

            inp_maxpref_i.resizeN(current_n);
            inp_maxpref_v.resizeN(current_n);
            inp_sum.resizeN(current_n);
            out_maxpref_i.resizeN(current_n);
            out_maxpref_v.resizeN(current_n);
            out_sum.resizeN(current_n);

            std::vector<unsigned int> inp_maxpref_i_init(current_n);
            for (unsigned i = 0; i < current_n; ++i) inp_maxpref_i_init[i] = i + 1;
            std::vector<int> garbage_init(current_n, 0);
            std::vector<unsigned> garbage_uinit(current_n, 0);

            inp_maxpref_i.writeN(garbage_uinit.data(), current_n);
            inp_maxpref_v.writeN(garbage_init.data(), current_n);
            inp_sum.writeN(garbage_init.data(), current_n);
            out_maxpref_i.writeN(garbage_uinit.data(), current_n);
            out_maxpref_v.writeN(garbage_init.data(), current_n);
            out_sum.writeN(garbage_init.data(), current_n);

            const_maxpref_i.resizeN(current_n);
            const_maxpref_v.resizeN(current_n);
            const_sum.resizeN(current_n);

            const_maxpref_i.writeN(inp_maxpref_i_init.data(), current_n);
            const_maxpref_v.writeN(as.data(), current_n);
            const_sum.writeN(as.data(), current_n);

            std::cout << "sizeof(int) = " << sizeof(int) << "\n";

            auto debug_print = [&]() {
                std::vector<unsigned> debug_maxpref_i(current_n);
                std::vector<int> debug_maxpref_v(current_n);
                std::vector<int> debug_sum(current_n);
                std::cout << "----------------------------\n";

                std::cout << "const\n";
                const_maxpref_i.readN(debug_maxpref_i.data(), current_n);
                const_maxpref_v.readN(debug_maxpref_v.data(), current_n);
                const_sum.readN(debug_sum.data(), current_n);
                for (unsigned i = 0; i < current_n; ++i) std::cout << debug_maxpref_i[i] << " "; std::cout << "\n";
                for (unsigned i = 0; i < current_n; ++i) std::cout << debug_maxpref_v[i] << " "; std::cout << "\n";
                for (unsigned i = 0; i < current_n; ++i) std::cout << debug_sum[i] << " "; std::cout << "\n";

                std::cout << "inp\n";
                inp_maxpref_i.readN(debug_maxpref_i.data(), current_n);
                inp_maxpref_v.readN(debug_maxpref_v.data(), current_n);
                inp_sum.readN(debug_sum.data(), current_n);
                for (unsigned i = 0; i < current_n; ++i) std::cout << debug_maxpref_i[i] << " "; std::cout << "\n";
                for (unsigned i = 0; i < current_n; ++i) std::cout << debug_maxpref_v[i] << " "; std::cout << "\n";
                for (unsigned i = 0; i < current_n; ++i) std::cout << debug_sum[i] << " "; std::cout << "\n";

                std::cout << "out\n";
                out_maxpref_i.readN(debug_maxpref_i.data(), current_n);
                out_maxpref_v.readN(debug_maxpref_v.data(), current_n);
                out_sum.readN(debug_sum.data(), current_n);
                for (unsigned i = 0; i < current_n; ++i) std::cout << debug_maxpref_i[i] << " "; std::cout << "\n";
                for (unsigned i = 0; i < current_n; ++i) std::cout << debug_maxpref_v[i] << " "; std::cout << "\n";
                for (unsigned i = 0; i < current_n; ++i) std::cout << debug_sum[i] << " "; std::cout << "\n";
            };

            timer t;

            for (int iBench = 0; iBench < 1; ++iBench) {
                #if !0
                bool firstRun = true;
                bool inp2out = true;

                current_n = n;
                next_n = gpu::divup(current_n, VALS_IN_STEP);

                while (current_n > 1) {
                    // debug_print();
                    if (firstRun) {
                        firstRun = false;
                        inp2out = false;
                        kernel.exec(gpu::WorkSize(WORKGROUP_SIZE, next_n),
                            const_maxpref_i, const_maxpref_v, const_sum,
                            out_maxpref_i, out_maxpref_v, out_sum,
                            current_n, next_n);
                    } else if (inp2out) {
                        inp2out = false;
                        kernel.exec(gpu::WorkSize(WORKGROUP_SIZE, next_n),
                            inp_maxpref_i, inp_maxpref_v, inp_sum,
                            out_maxpref_i, out_maxpref_v, out_sum,
                            current_n, next_n);
                    } else if (!inp2out) {
                        inp2out = true;
                        kernel.exec(gpu::WorkSize(WORKGROUP_SIZE, next_n),
                            out_maxpref_i, out_maxpref_v, out_sum,
                            inp_maxpref_i, inp_maxpref_v, inp_sum,
                            current_n, next_n);
                    }

                    current_n = next_n;
                    next_n = gpu::divup(current_n, VALS_IN_STEP);
                }

                // debug_print();
                inp2out = !inp2out;

                int max_sum;
                unsigned int result;
                if (firstRun) {
                    assert(n == 0 || n == 1);
                    if (n == 0) {
                        max_sum = 0;
                        result = 1;
                    } else if (n == 1) {
                        max_sum = as[0];
                        result = 2;
                    }
                } else if (inp2out) {
                    out_maxpref_v.readN(&max_sum, 1);
                    out_maxpref_i.readN(&result, 1);
                } else if (!inp2out) {
                    inp_maxpref_v.readN(&max_sum, 1);
                    inp_maxpref_i.readN(&result, 1);
                }
                #else
                auto res = myCPUpref(as);
                int max_sum = res.max_sum;
                unsigned int result = res.result;
                #endif

                EXPECT_THE_SAME(reference_max_sum, max_sum, "GPU result should be consistent!");
                EXPECT_THE_SAME(reference_result, result, "GPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
    }
}
