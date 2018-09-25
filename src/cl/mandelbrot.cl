#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void mandelbrot(__global float * kernel_res,
                         const float fromX, const float fromY,
                         const float sizeX, const float sizeY,
                         const unsigned int iterationsLimit)
{
    const float threshold = 256.0f;
    const float threshold2 = threshold * threshold;
    float x = fromX + sizeX * ((get_global_id(0) + 0.5f) / get_global_size(0)),
          y = fromY + sizeY * ((get_global_id(1) + 0.5f) / get_global_size(1));
    
    float x0 = x, y0 = y, xPrev;

    unsigned int iter = 0;
    for (; iter < iterationsLimit; ++iter) {
        xPrev = x;
        x = x * x - y * y + x0;
        y = 2.0f * xPrev * y + y0;
        if ((x * x + y * y) > threshold2) {
            break;
        }
    }

    kernel_res[get_global_id(1) * get_global_size(0) + get_global_id(0)] = (float)iter / iterationsLimit;
}
