/* Tiny synthetic E4M3FN tile primitive; this is not a model inference backend. */
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define FP8_EXPORT __declspec(dllexport)
#else
#define FP8_EXPORT
#endif

#define FP8_STRINGIFY_IMPL(value) #value
#define FP8_STRINGIFY(value) FP8_STRINGIFY_IMPL(value)
#if defined(_MSC_FULL_VER)
#define FP8_COMPILER "MSVC " FP8_STRINGIFY(_MSC_FULL_VER)
#else
#define FP8_COMPILER "non-MSVC C compiler"
#endif

#define FP8_MAX_TILE 128
#define FP8_OK 0
#define FP8_INVALID_ARGUMENT 1
#define FP8_NONFINITE 2

FP8_EXPORT int fp8_cpu_abi_version(void) { return 1; }

FP8_EXPORT const char *fp8_cpu_build_info(void)
{
    return FP8_COMPILER "; synthetic E4M3FN decode; FP32 sequential tile matvec; ABI 1";
}

FP8_EXPORT float fp8_cpu_decode_e4m3fn(uint8_t code)
{
    unsigned int magnitude = (unsigned int)code & 0x7fu;
    unsigned int exponent = magnitude >> 3;
    unsigned int mantissa = magnitude & 7u;
    float decoded;
    if (magnitude == 0x7fu) {
        return copysignf(NAN, (code & 0x80u) ? -1.0f : 1.0f);
    }
    if (exponent == 0u) {
        decoded = ldexpf((float)mantissa, -9);
    } else {
        decoded = ldexpf((float)(8u + mantissa), (int)exponent - 10);
    }
    return (code & 0x80u) ? -decoded : decoded;
}

/* The caller supplies bounded buffers. Lengths are element counts, not bytes. */
FP8_EXPORT int fp8_cpu_matvec_tile(
    const uint8_t *weights, size_t weight_count,
    int rows, int cols, const float *vector, size_t vector_count,
    float scale, float *output, size_t output_count)
{
    size_t count;
    size_t index;
    int row;
    int col;
    if (weights == NULL || vector == NULL || output == NULL ||
        rows < 1 || rows > FP8_MAX_TILE || cols < 1 || cols > FP8_MAX_TILE) {
        return FP8_INVALID_ARGUMENT;
    }
    count = (size_t)rows * (size_t)cols;
    if (weight_count != count || vector_count != (size_t)cols ||
        output_count != (size_t)rows || !isfinite(scale) || scale <= 0.0f) {
        return FP8_INVALID_ARGUMENT;
    }
    for (index = 0; index < count; ++index) {
        if ((weights[index] & 0x7fu) == 0x7fu) {
            return FP8_NONFINITE;
        }
    }
    for (col = 0; col < cols; ++col) {
        if (!isfinite(vector[col])) {
            return FP8_NONFINITE;
        }
    }
    for (row = 0; row < rows; ++row) {
        float sum = 0.0f;
        for (col = 0; col < cols; ++col) {
            float weight = fp8_cpu_decode_e4m3fn(weights[(size_t)row * (size_t)cols +
                                                       (size_t)col]);
            float scaled = weight * scale;
            float product = scaled * vector[col];
            sum = sum + product;
            if (!isfinite(scaled) || !isfinite(product) || !isfinite(sum)) {
                return FP8_NONFINITE;
            }
        }
        output[row] = sum;
    }
    return FP8_OK;
}
