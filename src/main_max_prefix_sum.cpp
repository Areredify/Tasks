#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include <conio.h>


#include "cl/max_prefix_sum_cl.h"


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


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
            as[i] = (unsigned int) r.next(-values_range, values_range);
        }

        int reference_max_sum;
        int reference_result;
        {
            int max_sum = 0;
            int sum = 0;
            int result = 0;
            for (int i = 0; i < n; ++i) {
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
                int max_sum = 0;
                int sum = 0;
                int result = 0;
                for (int i = 0; i < n; ++i) {
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
            //gpu::Device device = gpu::chooseGPUDevice(argc, argv);
            char *argvv[] = { "poop", "0" };
            gpu::Device device = gpu::chooseGPUDevice(2, argvv);

            gpu::Context context;
            context.init(device.device_id_opencl);
            context.activate();
            {
                ocl::Kernel kernel(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "max_prefix_sum");

                bool printLog = false;
                kernel.compile(printLog);
                gpu::gpu_mem_32i sum_input, sum_output, pref_input, pref_output;
                sum_input.resizeN(n), sum_output.resizeN(n);
                pref_input.resizeN(n), pref_output.resizeN(n);

                unsigned int work_group_size = 128;
                std::vector<int> kernel_sum_input = as, kernel_pref_input = as, buf_sum, buf_pref;
                for (auto & i : kernel_pref_input)
                    i = std::max(i, 0);

                buf_sum = kernel_sum_input, buf_pref = kernel_pref_input;
                std::cout << std::endl;
                timer t;
                for (int iter = 0; iter < benchmarkingIters; ++iter) {
                    t.stop();
                    int size = n;
                    kernel_sum_input = buf_sum;
                    kernel_pref_input = buf_pref;
                    t.start();
                    while (size > 1) {
                        sum_input.writeN(kernel_sum_input.data(), size);
                        pref_input.writeN(kernel_pref_input.data(), size);

                        kernel.exec(gpu::WorkSize(work_group_size, (size + work_group_size - 1) / work_group_size * work_group_size),
                                    sum_input, pref_input, size, sum_output, pref_output);

                        size = (size + work_group_size - 1) / work_group_size;
                        kernel_sum_input.resize(size);
                        kernel_pref_input.resize(size);
                        sum_output.readN(kernel_sum_input.data(), size);
                        pref_output.readN(kernel_pref_input.data(), size);
                    }
                    EXPECT_THE_SAME(kernel_pref_input[0], reference_max_sum, "GPU result should be consistent!");
                    t.nextLap();
                }

                std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
                std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
            }
        }
    }

    _getch();
    return 0;
}
