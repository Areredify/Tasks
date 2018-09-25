#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

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
    int benchmarkingIters = 11;
    int max_n = 16 * (1 << 24);

    gpu::Device device = gpu::chooseGPUDevice(argc, argv);
    //char *argvv[] = { "poop", "0" };
    //gpu::Device device = gpu::chooseGPUDevice(2, argvv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

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
            ocl::Kernel kernel(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "max_prefix_sum");

            bool printLog = false;
            kernel.compile(printLog);
            gpu::gpu_mem_32i bufs[6];
            for (auto & i : bufs)
                i.resizeN(n);

            unsigned int work_group_size = 128;
            std::vector<int> kernel_sum_input = as, kernel_pref_input = as, indexes(n);
            for (auto & i : kernel_pref_input)
                i = std::max(i, 0);

            for (int i = 0; i < n; i++) {
                if (as[i] >= 0)
                    indexes[i] = i + 1;
                else
                    indexes[i] = i;
            }

            int curs = 0;
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                t.stop();
                int size = n;
                bufs[curs].writeN(kernel_sum_input.data(), n);
                bufs[curs + 1].writeN(kernel_pref_input.data(), n);
                bufs[curs + 2].writeN(indexes.data(), n);
                t.start();
                while (size > 1) {
                    kernel.exec(gpu::WorkSize(work_group_size, (size + work_group_size - 1) / work_group_size * work_group_size),
                                bufs[curs], bufs[curs + 1], bufs[curs + 2], size, bufs[3 - curs], bufs[3 - curs + 1], bufs[3 - curs + 2]);

                    size = (size + work_group_size - 1) / work_group_size;
                    curs = 3 - curs;
                }
                int res1, res2;
                bufs[curs + 1].readN(&res1, 1);
                bufs[curs + 2].readN(&res2, 1);

                EXPECT_THE_SAME(res1, reference_max_sum, "GPU result should be consistent!");
                EXPECT_THE_SAME(res2, reference_result, "GPU result should be consistent!");
                t.nextLap();
            }

            std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
    }

    return 0;
}
