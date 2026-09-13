/* Tiny synthetic E4M3FN tile primitive; this is not a model inference backend. */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <float.h>
#include <xmmintrin.h>
#include "cpu_row_pool.h"

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

FP8_EXPORT void *fp8_cpu_row_pool_create(int threads) { return cpu_rows_pool_create(threads); }
FP8_EXPORT void fp8_cpu_row_pool_destroy(void *pool) { cpu_rows_pool_destroy((cpu_rows_pool *)pool); }

typedef struct dense_row_job {
    const uint8_t *weights;
    const float *vector;
    float *output;
    int cols;
    int dtype;
    size_t itemsize;
} dense_row_job;

/* For small input batches, independent rows fill SIMD lanes instead. The
 * final0..3rows remain on the original scalar path; no reduction is reordered. */
static int dense_four_rows(const uint8_t *weights, const float *vectors, float *output,
                          int cols, int batch, int dtype, size_t itemsize, int start, int stop)
{
    const __m128 sign = _mm_set1_ps(-0.0f), limit = _mm_set1_ps(FLT_MAX);
    int row, tile, col, lane, offset;
    for (row = start; row + 4 <= stop; row += 4) {
        __m128 total[3];
        for (lane = 0; lane < batch; ++lane) { total[lane] = _mm_setzero_ps(); }
        for (tile = 0; tile < cols; tile += 128) {
            int end = tile + 128 < cols ? tile + 128 : cols;
            __m128 partial[3];
            for (lane = 0; lane < batch; ++lane) { partial[lane] = _mm_setzero_ps(); }
            for (col = tile; col < end; ++col) {
                float w0 = fp8_cpu_decode_dense(weights + ((size_t)row * (size_t)cols + (size_t)col) * itemsize, dtype);
                float w1 = fp8_cpu_decode_dense(weights + ((size_t)(row + 1) * (size_t)cols + (size_t)col) * itemsize, dtype);
                float w2 = fp8_cpu_decode_dense(weights + ((size_t)(row + 2) * (size_t)cols + (size_t)col) * itemsize, dtype);
                float w3 = fp8_cpu_decode_dense(weights + ((size_t)(row + 3) * (size_t)cols + (size_t)col) * itemsize, dtype);
                __m128 weight = _mm_set_ps(w3, w2, w1, w0);
                if (_mm_movemask_ps(_mm_cmple_ps(_mm_andnot_ps(sign, weight), limit)) != 15) { return FP8_NONFINITE; }
                for (lane = 0; lane < batch; ++lane) {
                    __m128 product = _mm_mul_ps(weight, _mm_set1_ps(vectors[(size_t)col * (size_t)batch + (size_t)lane]));
                    partial[lane] = _mm_add_ps(partial[lane], product);
                    if (_mm_movemask_ps(_mm_cmple_ps(_mm_andnot_ps(sign, partial[lane]), limit)) != 15) { return FP8_NONFINITE; }
                }
            }
            for (lane = 0; lane < batch; ++lane) {
                total[lane] = _mm_add_ps(total[lane], partial[lane]);
                if (_mm_movemask_ps(_mm_cmple_ps(_mm_andnot_ps(sign, total[lane]), limit)) != 15) { return FP8_NONFINITE; }
            }
        }
        for (lane = 0; lane < batch; ++lane) {
            float values[4];
            _mm_storeu_ps(values, total[lane]);
            for (offset = 0; offset < 4; ++offset) { output[(size_t)(row + offset) * (size_t)batch + (size_t)lane] = values[offset]; }
        }
    }
    return FP8_OK;
}

static int dense_rows(void *argument, int start, int stop)
{
    dense_row_job *job = (dense_row_job *)argument;
    int row, tile, col;
    int status = dense_four_rows(job->weights, job->vector, job->output, job->cols, 1,
                                job->dtype, job->itemsize, start, stop);
    if (status != FP8_OK) { return status; }
    start += ((stop - start) / 4) * 4;
    for (row = start; row < stop; ++row) {
        float total = 0.0f;
        for (tile = 0; tile < job->cols; tile += 128) {
            int end = tile + 128 < job->cols ? tile + 128 : job->cols;
            float partial = 0.0f;
            for (col = tile; col < end; ++col) {
                float weight = fp8_cpu_decode_dense(job->weights + ((size_t)row * (size_t)job->cols + (size_t)col) * job->itemsize, job->dtype);
                float product = weight * job->vector[col];
                partial = partial + product;
                if (!isfinite(weight) || !isfinite(product) || !isfinite(partial)) { return FP8_NONFINITE; }
            }
            total = total + partial;
            if (!isfinite(total)) { return FP8_NONFINITE; }
        }
        job->output[row] = total;
    }
    return FP8_OK;
}

FP8_EXPORT int fp8_cpu_matvec_dense_row_band(
    const uint8_t *weights, size_t weight_bytes, int rows, int cols, int dtype,
    const float *vector, size_t vector_count, float *output, size_t output_count, void *row_pool)
{
    dense_row_job job;
    int col;
    size_t itemsize;
    if (weights == NULL || vector == NULL || output == NULL || rows < 1 || rows > 128 ||
        cols < 1 || cols > 16384 || dtype < 1 || dtype > 3) { return FP8_INVALID_ARGUMENT; }
    itemsize = dtype == 3 ? 4u : 2u;
    if (weight_bytes != (size_t)rows * (size_t)cols * itemsize || vector_count != (size_t)cols || output_count != (size_t)rows) {
        return FP8_INVALID_ARGUMENT;
    }
    for (col = 0; col < cols; ++col) { if (!isfinite(vector[col])) { return FP8_NONFINITE; } }
    job.weights = weights;
    job.vector = vector;
    job.output = output;
    job.cols = cols;
    job.dtype = dtype;
    job.itemsize = itemsize;
    return cpu_rows_run((cpu_rows_pool *)row_pool, dense_rows, &job, rows, rows * cols >= 65536);
}

/* Independent vectors occupy SIMD lanes; no horizontal reduction or FMA.
 * Every lane still sums columns sequentially in128-column tiles. */
typedef struct dense_many_job {
    const uint8_t *weights;
    const float *vectors;
    float *output;
    int cols, batch, dtype;
    size_t itemsize;
} dense_many_job;

static int dense_many_rows(void *argument, int start, int stop)
{
    dense_many_job *job = (dense_many_job *)argument;
    int row, tile, col, lane, group;
    int groups = job->batch / 4, first_tail = groups * 4;
    const __m128 sign = _mm_set1_ps(-0.0f), limit = _mm_set1_ps(FLT_MAX);
    if (job->batch <= 3) {
        int status = dense_four_rows(job->weights, job->vectors, job->output, job->cols, job->batch,
                                    job->dtype, job->itemsize, start, stop);
        if (status != FP8_OK) { return status; }
        start += ((stop - start) / 4) * 4;
    }
    for (row = start; row < stop; ++row) {
        __m128 totals[4];
        float tail_total[3] = {0.0f, 0.0f, 0.0f};
        for (group = 0; group < groups; ++group) { totals[group] = _mm_setzero_ps(); }
        for (tile = 0; tile < job->cols; tile += 128) {
            int end = tile + 128 < job->cols ? tile + 128 : job->cols;
            __m128 partials[4];
            float tail_partial[3] = {0.0f, 0.0f, 0.0f};
            for (group = 0; group < groups; ++group) { partials[group] = _mm_setzero_ps(); }
            for (col = tile; col < end; ++col) {
                float weight = fp8_cpu_decode_dense(job->weights + ((size_t)row * (size_t)job->cols + (size_t)col) * job->itemsize, job->dtype);
                const float *input = job->vectors + (size_t)col * (size_t)job->batch;
                __m128 broadcast;
                if (!isfinite(weight)) { return FP8_NONFINITE; }
                broadcast = _mm_set1_ps(weight);
                for (group = 0; group < groups; ++group) {
                    __m128 product = _mm_mul_ps(broadcast, _mm_loadu_ps(input + 4 * group));
                    partials[group] = _mm_add_ps(partials[group], product);
                    if (_mm_movemask_ps(_mm_cmple_ps(_mm_andnot_ps(sign, partials[group]), limit)) != 15) { return FP8_NONFINITE; }
                }
                for (lane = first_tail; lane < job->batch; ++lane) {
                    float product = weight * input[lane];
                    tail_partial[lane - first_tail] = tail_partial[lane - first_tail] + product;
                    if (!isfinite(tail_partial[lane - first_tail])) { return FP8_NONFINITE; }
                }
            }
            for (group = 0; group < groups; ++group) {
                totals[group] = _mm_add_ps(totals[group], partials[group]);
                if (_mm_movemask_ps(_mm_cmple_ps(_mm_andnot_ps(sign, totals[group]), limit)) != 15) { return FP8_NONFINITE; }
            }
            for (lane = first_tail; lane < job->batch; ++lane) {
                tail_total[lane - first_tail] = tail_total[lane - first_tail] + tail_partial[lane - first_tail];
                if (!isfinite(tail_total[lane - first_tail])) { return FP8_NONFINITE; }
            }
        }
        for (group = 0; group < groups; ++group) { _mm_storeu_ps(job->output + (size_t)row * (size_t)job->batch + 4 * group, totals[group]); }
        for (lane = first_tail; lane < job->batch; ++lane) { job->output[(size_t)row * (size_t)job->batch + (size_t)lane] = tail_total[lane - first_tail]; }
    }
    return FP8_OK;
}

FP8_EXPORT int fp8_cpu_matvec_dense_row_band_many(
    const uint8_t *weights, size_t weight_bytes, int rows, int cols, int batch, int dtype,
    const float *vectors, size_t vector_count, float *output, size_t output_count, void *row_pool)
{
    dense_many_job job;
    size_t index, itemsize;
    if (weights == NULL || vectors == NULL || output == NULL || rows < 1 || rows > 128 ||
        cols < 1 || cols > 16384 || batch < 1 || batch > 16 || dtype < 1 || dtype > 3) { return FP8_INVALID_ARGUMENT; }
    itemsize = dtype == 3 ? 4u : 2u;
    if (weight_bytes != (size_t)rows * (size_t)cols * itemsize || vector_count != (size_t)cols * (size_t)batch ||
        output_count != (size_t)rows * (size_t)batch) { return FP8_INVALID_ARGUMENT; }
    for (index = 0; index < vector_count; ++index) { if (!isfinite(vectors[index])) { return FP8_NONFINITE; } }
    job.weights = weights; job.vectors = vectors; job.output = output;
    job.cols = cols; job.batch = batch; job.dtype = dtype; job.itemsize = itemsize;
    return cpu_rows_run((cpu_rows_pool *)row_pool, dense_many_rows, &job, rows, rows * cols * batch >= 65536);
}
