/* Tiny synthetic E4M3FN tile primitive; this is not a model inference backend. */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

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
    return FP8_COMPILER "; E4M3FN and BF16/F16/F32 decode; FP32 sequential tile matvec; ABI 1";
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

/* Additive ABI 1 entry point. dtype: 1=BF16, 2=IEEE binary16, 3=FP32.
 * Decode little-endian storage explicitly; do not require aligned caller bytes. */
static float fp8_cpu_decode_dense(const uint8_t *bytes, int dtype)
{
    uint32_t bits;
    float result;
    if (dtype == 2) {
        unsigned int code = (unsigned int)bytes[0] | ((unsigned int)bytes[1] << 8);
        unsigned int exponent = (code >> 10) & 31u;
        unsigned int fraction = code & 1023u;
        if (exponent == 31u) {
            result = fraction == 0u ? INFINITY : NAN;
        } else {
            result = exponent == 0u ? ldexpf((float)fraction, -24)
                                    : ldexpf((float)(1024u + fraction), (int)exponent - 25);
        }
        return (code & 32768u) != 0u ? -result : result;
    }
    bits = dtype == 1 ? ((uint32_t)bytes[0] << 16) | ((uint32_t)bytes[1] << 24)
        : (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) | ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
    memcpy(&result, &bits, sizeof(result));
    return result;
}

FP8_EXPORT int fp8_cpu_matvec_dense_tile(
    const uint8_t *weights, size_t weight_bytes, int rows, int cols, int dtype,
    const float *vector, size_t vector_count, float *output, size_t output_count)
{
    int row, col;
    size_t index, count, itemsize;
    if (weights == NULL || vector == NULL || output == NULL ||
        rows < 1 || rows > FP8_MAX_TILE || cols < 1 || cols > FP8_MAX_TILE || dtype < 1 || dtype > 3) {
        return FP8_INVALID_ARGUMENT;
    }
    itemsize = dtype == 3 ? 4u : 2u;
    count = (size_t)rows * (size_t)cols;
    if (weight_bytes != count * itemsize || vector_count != (size_t)cols || output_count != (size_t)rows) {
        return FP8_INVALID_ARGUMENT;
    }
    /* Match the scalar reader's all-weight finite check before computation. */
    for (index = 0; index < count; ++index) {
        if (!isfinite(fp8_cpu_decode_dense(weights + index * itemsize, dtype))) {
            return FP8_NONFINITE;
        }
    }
    for (col = 0; col < cols; ++col) {
        if (!isfinite(vector[col])) { return FP8_NONFINITE; }
    }
    for (row = 0; row < rows; ++row) {
        float sum = 0.0f;
        for (col = 0; col < cols; ++col) {
            float weight = fp8_cpu_decode_dense(weights + ((size_t)row * (size_t)cols + (size_t)col) * itemsize, dtype);
            float product = weight * vector[col];
            sum = sum + product;
            if (!isfinite(product) || !isfinite(sum)) { return FP8_NONFINITE; }
        }
        output[row] = sum;
    }
    return FP8_OK;
}
