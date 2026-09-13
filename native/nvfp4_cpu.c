/* Bounded ModelOpt NVFP4 decoded-weight reference. Inputs are not quantized.
 * Layout: NVIDIA ModelOpt 51de53e48ccae8804f8fe1198b7cf89475c5c4f4,
 * modelopt/torch/quantization/qtensor/nvfp4_tensor.py. */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <float.h>
#include <xmmintrin.h>
#include "cpu_row_pool.h"

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

/* Additive ABI1 entry point. A bounded encoded row band batches the Python
 * calls, but each row still reduces128-column tiles independently and adds
 * those partials in order, exactly as the original tile executor does. */
NVFP4_EXPORT int nvfp4_cpu_matvec_row_band(
    const uint8_t *packed, size_t packed_count, int rows, int cols,
    const uint8_t *scales, size_t scale_count,
    const float *vector, size_t vector_count, float global_scale,
    float *output, size_t output_count)
{
    size_t index;
    int row, col, tile, group;
    if (packed == NULL || scales == NULL || vector == NULL || output == NULL ||
        rows < 1 || rows > 128 || cols < 16 || cols > 16384 || cols % 16 != 0 ||
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
        float total = 0.0f;
        for (tile = 0; tile < cols; tile += 128) {
            int stop = tile + 128 < cols ? tile + 128 : cols;
            float partial = 0.0f;
            for (group = tile; group < stop; group += 16) {
                float combined = decode_scale(scales[(size_t)row * (size_t)(cols / 16) + (size_t)(group / 16)]) * global_scale;
                if (!isfinite(combined)) { return 2; }
                for (col = group; col < group + 16; ++col) {
                    uint8_t byte = packed[(size_t)row * (size_t)(cols / 2) + (size_t)(col / 2)];
                    unsigned int code = ((unsigned int)byte >> ((col % 2) * 4)) & 15u;
                    float weight = decode_e2m1(code) * combined;
                    float product = weight * vector[col];
                    partial = partial + product;
                    if (!isfinite(weight) || !isfinite(product) || !isfinite(partial)) { return 2; }
                }
            }
            total = total + partial;
            if (!isfinite(total)) { return 2; }
        }
        output[row] = total;
    }
    return 0;
}

NVFP4_EXPORT void *nvfp4_cpu_row_pool_create(int threads) { return cpu_rows_pool_create(threads); }
NVFP4_EXPORT void nvfp4_cpu_row_pool_destroy(void *pool) { cpu_rows_pool_destroy((cpu_rows_pool *)pool); }

typedef struct nvfp4_row_job {
    const uint8_t *packed;
    const uint8_t *scales;
    const float *vector;
    float *output;
    int cols;
    float global_scale;
} nvfp4_row_job;

static int nvfp4_four_rows(const uint8_t *packed, const uint8_t *scales, const float *vectors,
                          float *output, int cols, int batch, float global_scale, int start, int stop)
{
    const __m128 sign = _mm_set1_ps(-0.0f), limit = _mm_set1_ps(FLT_MAX);
    int row, tile, block, col, lane, offset;
    for (row = start; row + 4 <= stop; row += 4) {
        __m128 total[3];
        for (lane = 0; lane < batch; ++lane) { total[lane] = _mm_setzero_ps(); }
        for (tile = 0; tile < cols; tile += 128) {
            int end = tile + 128 < cols ? tile + 128 : cols;
            __m128 partial[3];
            for (lane = 0; lane < batch; ++lane) { partial[lane] = _mm_setzero_ps(); }
            for (block = tile; block < end; block += 16) {
                float scale0 = decode_scale(scales[(size_t)row * (size_t)(cols / 16) + (size_t)(block / 16)]) * global_scale;
                float scale1 = decode_scale(scales[(size_t)(row + 1) * (size_t)(cols / 16) + (size_t)(block / 16)]) * global_scale;
                float scale2 = decode_scale(scales[(size_t)(row + 2) * (size_t)(cols / 16) + (size_t)(block / 16)]) * global_scale;
                float scale3 = decode_scale(scales[(size_t)(row + 3) * (size_t)(cols / 16) + (size_t)(block / 16)]) * global_scale;
                __m128 combined = _mm_set_ps(scale3, scale2, scale1, scale0);
                if (_mm_movemask_ps(_mm_cmple_ps(_mm_andnot_ps(sign, combined), limit)) != 15) { return 2; }
                for (col = block; col < block + 16; ++col) {
                    unsigned int shift = (unsigned int)(col % 2) * 4u;
                    float w0 = decode_e2m1(((unsigned int)packed[(size_t)row * (size_t)(cols / 2) + (size_t)(col / 2)] >> shift) & 15u);
                    float w1 = decode_e2m1(((unsigned int)packed[(size_t)(row + 1) * (size_t)(cols / 2) + (size_t)(col / 2)] >> shift) & 15u);
                    float w2 = decode_e2m1(((unsigned int)packed[(size_t)(row + 2) * (size_t)(cols / 2) + (size_t)(col / 2)] >> shift) & 15u);
                    float w3 = decode_e2m1(((unsigned int)packed[(size_t)(row + 3) * (size_t)(cols / 2) + (size_t)(col / 2)] >> shift) & 15u);
                    __m128 weight = _mm_mul_ps(_mm_set_ps(w3, w2, w1, w0), combined);
                    if (_mm_movemask_ps(_mm_cmple_ps(_mm_andnot_ps(sign, weight), limit)) != 15) { return 2; }
                    for (lane = 0; lane < batch; ++lane) {
                        __m128 product = _mm_mul_ps(weight, _mm_set1_ps(vectors[(size_t)col * (size_t)batch + (size_t)lane]));
                        partial[lane] = _mm_add_ps(partial[lane], product);
                        if (_mm_movemask_ps(_mm_cmple_ps(_mm_andnot_ps(sign, partial[lane]), limit)) != 15) { return 2; }
                    }
                }
            }
            for (lane = 0; lane < batch; ++lane) {
                total[lane] = _mm_add_ps(total[lane], partial[lane]);
                if (_mm_movemask_ps(_mm_cmple_ps(_mm_andnot_ps(sign, total[lane]), limit)) != 15) { return 2; }
            }
        }
        for (lane = 0; lane < batch; ++lane) {
            float values[4];
            _mm_storeu_ps(values, total[lane]);
            for (offset = 0; offset < 4; ++offset) { output[(size_t)(row + offset) * (size_t)batch + (size_t)lane] = values[offset]; }
        }
    }
    return 0;
}

static int nvfp4_rows(void *argument, int start, int stop)
{
    nvfp4_row_job *job = (nvfp4_row_job *)argument;
    int row, tile, group, col;
    int status = nvfp4_four_rows(job->packed, job->scales, job->vector, job->output,
                               job->cols, 1, job->global_scale, start, stop);
    if (status != 0) { return status; }
    start += ((stop - start) / 4) * 4;
    for (row = start; row < stop; ++row) {
        float total = 0.0f;
        for (tile = 0; tile < job->cols; tile += 128) {
            int end = tile + 128 < job->cols ? tile + 128 : job->cols;
            float partial = 0.0f;
            for (group = tile; group < end; group += 16) {
                float combined = decode_scale(job->scales[(size_t)row * (size_t)(job->cols / 16) + (size_t)(group / 16)]) * job->global_scale;
                if (!isfinite(combined)) { return 2; }
                for (col = group; col < group + 16; ++col) {
                    uint8_t byte = job->packed[(size_t)row * (size_t)(job->cols / 2) + (size_t)(col / 2)];
                    unsigned int code = ((unsigned int)byte >> ((col % 2) * 4)) & 15u;
                    float weight = decode_e2m1(code) * combined;
                    float product = weight * job->vector[col];
                    partial = partial + product;
                    if (!isfinite(weight) || !isfinite(product) || !isfinite(partial)) { return 2; }
                }
            }
            total = total + partial;
            if (!isfinite(total)) { return 2; }
        }
        job->output[row] = total;
    }
    return 0;
}

/* Original row-band ABI stays serial and unchanged; this additive entry point
 * accepts an explicitly owned, bounded worker pool as its final parameter. */
NVFP4_EXPORT int nvfp4_cpu_matvec_row_band_parallel(
    const uint8_t *packed, size_t packed_count, int rows, int cols,
    const uint8_t *scales, size_t scale_count,
    const float *vector, size_t vector_count, float global_scale,
    float *output, size_t output_count, void *row_pool)
{
    nvfp4_row_job job;
    size_t index;
    int col;
    if (packed == NULL || scales == NULL || vector == NULL || output == NULL || rows < 1 || rows > 128 ||
        cols < 16 || cols > 16384 || cols % 16 != 0 || !isfinite(global_scale) || global_scale <= 0.0f) { return 1; }
    if (packed_count != (size_t)rows * (size_t)(cols / 2) || scale_count != (size_t)rows * (size_t)(cols / 16) ||
        vector_count != (size_t)cols || output_count != (size_t)rows) { return 1; }
    for (index = 0; index < scale_count; ++index) {
        if ((scales[index] & 0x7fu) == 0x7fu) { return 2; }
        if ((scales[index] & 0x80u) != 0u) { return 1; }
    }
    for (col = 0; col < cols; ++col) { if (!isfinite(vector[col])) { return 2; } }
    job.packed = packed;
    job.scales = scales;
    job.vector = vector;
    job.output = output;
    job.cols = cols;
    job.global_scale = global_scale;
    return cpu_rows_run((cpu_rows_pool *)row_pool, nvfp4_rows, &job, rows, rows * cols >= 65536);
}

typedef struct nvfp4_many_job {
    const uint8_t *packed, *scales;
    const float *vectors;
    float *output;
    int cols, batch;
    float global_scale;
} nvfp4_many_job;

static int nvfp4_many_rows(void *argument, int start, int stop)
{
    nvfp4_many_job *job = (nvfp4_many_job *)argument;
    int row, tile, block, col, group, lane;
    int groups = job->batch / 4, first_tail = groups * 4;
    const __m128 sign = _mm_set1_ps(-0.0f), limit = _mm_set1_ps(FLT_MAX);
    if (job->batch <= 3) {
        int status = nvfp4_four_rows(job->packed, job->scales, job->vectors, job->output,
                                   job->cols, job->batch, job->global_scale, start, stop);
        if (status != 0) { return status; }
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
            for (block = tile; block < end; block += 16) {
                float combined = decode_scale(job->scales[(size_t)row * (size_t)(job->cols / 16) + (size_t)(block / 16)]) * job->global_scale;
                if (!isfinite(combined)) { return 2; }
                for (col = block; col < block + 16; ++col) {
                    uint8_t byte = job->packed[(size_t)row * (size_t)(job->cols / 2) + (size_t)(col / 2)];
                    unsigned int code = ((unsigned int)byte >> ((col % 2) * 4)) & 15u;
                    float weight = decode_e2m1(code) * combined;
                    const float *input = job->vectors + (size_t)col * (size_t)job->batch;
                    __m128 broadcast;
                    if (!isfinite(weight)) { return 2; }
                    broadcast = _mm_set1_ps(weight);
                    for (group = 0; group < groups; ++group) {
                        __m128 product = _mm_mul_ps(broadcast, _mm_loadu_ps(input + group * 4));
                        partials[group] = _mm_add_ps(partials[group], product);
                        if (_mm_movemask_ps(_mm_cmple_ps(_mm_andnot_ps(sign, partials[group]), limit)) != 15) { return 2; }
                    }
                    for (lane = first_tail; lane < job->batch; ++lane) {
                        float product = weight * input[lane];
                        tail_partial[lane - first_tail] = tail_partial[lane - first_tail] + product;
                        if (!isfinite(tail_partial[lane - first_tail])) { return 2; }
                    }
                }
            }
            for (group = 0; group < groups; ++group) {
                totals[group] = _mm_add_ps(totals[group], partials[group]);
                if (_mm_movemask_ps(_mm_cmple_ps(_mm_andnot_ps(sign, totals[group]), limit)) != 15) { return 2; }
            }
            for (lane = first_tail; lane < job->batch; ++lane) {
                tail_total[lane - first_tail] = tail_total[lane - first_tail] + tail_partial[lane - first_tail];
                if (!isfinite(tail_total[lane - first_tail])) { return 2; }
            }
        }
        for (group = 0; group < groups; ++group) { _mm_storeu_ps(job->output + (size_t)row * (size_t)job->batch + group * 4, totals[group]); }
        for (lane = first_tail; lane < job->batch; ++lane) { job->output[(size_t)row * (size_t)job->batch + (size_t)lane] = tail_total[lane - first_tail]; }
    }
    return 0;
}

/* ABI1 additive: exact interleaved [cols,batch] input and [rows,batch] output. */
NVFP4_EXPORT int nvfp4_cpu_matvec_row_band_many(
    const uint8_t *packed, size_t packed_count, int rows, int cols, int batch,
    const uint8_t *scales, size_t scale_count, const float *vectors, size_t vector_count,
    float global_scale, float *output, size_t output_count, void *row_pool)
{
    nvfp4_many_job job;
    size_t index;
    if (packed == NULL || scales == NULL || vectors == NULL || output == NULL || rows < 1 || rows > 128 ||
        cols < 16 || cols > 16384 || cols % 16 != 0 || batch < 1 || batch > 16 ||
        !isfinite(global_scale) || global_scale <= 0.0f) { return 1; }
    if (packed_count != (size_t)rows * (size_t)(cols / 2) || scale_count != (size_t)rows * (size_t)(cols / 16) ||
        vector_count != (size_t)cols * (size_t)batch || output_count != (size_t)rows * (size_t)batch) { return 1; }
    for (index = 0; index < scale_count; ++index) {
        if ((scales[index] & 0x7fu) == 0x7fu) { return 2; }
        if ((scales[index] & 0x80u) != 0u) { return 1; }
    }
    for (index = 0; index < vector_count; ++index) { if (!isfinite(vectors[index])) { return 2; } }
    job.packed = packed; job.scales = scales; job.vectors = vectors; job.output = output;
    job.cols = cols; job.batch = batch; job.global_scale = global_scale;
    return cpu_rows_run((cpu_rows_pool *)row_pool, nvfp4_many_rows, &job, rows, rows * cols * batch >= 65536);
}
