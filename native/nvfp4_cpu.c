/* Bounded ModelOpt NVFP4 decoded-weight reference. Inputs are not quantized.
 * Layout: NVIDIA ModelOpt 51de53e48ccae8804f8fe1198b7cf89475c5c4f4,
 * modelopt/torch/quantization/qtensor/nvfp4_tensor.py. */
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define NVFP4_EXPORT __declspec(dllexport)
#else
#define NVFP4_EXPORT
#endif
#define NVFP4_STRINGIFY_IMPL(value) #value
#define NVFP4_STRINGIFY(value) NVFP4_STRINGIFY_IMPL(value)
#if defined(_MSC_FULL_VER)
#define NVFP4_COMPILER "MSVC " NVFP4_STRINGIFY(_MSC_FULL_VER)
#else
#define NVFP4_COMPILER "non-MSVC C compiler"
#endif

NVFP4_EXPORT int nvfp4_cpu_abi_version(void) { return 1; }
NVFP4_EXPORT const char *nvfp4_cpu_build_info(void)
{
    return NVFP4_COMPILER "; NVFP4 decoded-weight FP32 sequential matvec; activation quantization none; ABI 1";
}

static float decode_scale(uint8_t code)
{
    unsigned int exponent = (unsigned int)code >> 3;
    unsigned int mantissa = (unsigned int)code & 7u;
    return exponent == 0u ? ldexpf((float)mantissa, -9)
                          : ldexpf((float)(8u + mantissa), (int)exponent - 10);
}

static float decode_e2m1(unsigned int code)
{
    static const float values[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    return values[code];
}

/* Byte lengths and vector/output element counts are independently checked.
 * Return 0 success, 1 invalid argument, 2 nonfinite arithmetic/data. */
NVFP4_EXPORT int nvfp4_cpu_matvec_tile(
    const uint8_t *packed, size_t packed_count, int rows, int cols,
    const uint8_t *scales, size_t scale_count,
    const float *vector, size_t vector_count, float global_scale,
    float *output, size_t output_count)
{
    size_t index;
    int row, col;
    if (packed == NULL || scales == NULL || vector == NULL || output == NULL ||
        rows < 1 || rows > 128 || cols < 16 || cols > 128 || cols % 16 != 0 ||
        !isfinite(global_scale) || global_scale <= 0.0f) {
        return 1;
    }
    if (packed_count != (size_t)rows * (size_t)(cols / 2) ||
        scale_count != (size_t)rows * (size_t)(cols / 16) ||
        vector_count != (size_t)cols || output_count != (size_t)rows) {
        return 1;
    }
    for (index = 0; index < scale_count; ++index) {
        if ((scales[index] & 0x7fu) == 0x7fu) { return 2; }
        if ((scales[index] & 0x80u) != 0u) { return 1; }
    }
    for (col = 0; col < cols; ++col) {
        if (!isfinite(vector[col])) { return 2; }
    }
    for (row = 0; row < rows; ++row) {
        float sum = 0.0f;
        for (col = 0; col < cols; ++col) {
            uint8_t byte = packed[(size_t)row * (size_t)(cols / 2) + (size_t)(col / 2)];
            unsigned int code = ((unsigned int)byte >> ((col % 2) * 4)) & 15u;
            float scale = decode_scale(scales[(size_t)row * (size_t)(cols / 16) + (size_t)(col / 16)]);
            float combined_scale = scale * global_scale;
            float weight = decode_e2m1(code) * combined_scale;
            float product = weight * vector[col];
            sum = sum + product;
            if (!isfinite(combined_scale) || !isfinite(weight) || !isfinite(product) || !isfinite(sum)) {
                return 2;
            }
        }
        output[row] = sum;
    }
    return 0;
}
