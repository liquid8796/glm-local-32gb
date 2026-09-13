/* Persistent per-backend row workers. The calling thread participates, so a
 * configured count of8 means at most7 pool workers plus the caller. Jobs only
 * split independent output rows; each row's arithmetic order is unchanged. */
#ifndef MODEL_DESK_CPU_ROW_POOL_H
#define MODEL_DESK_CPU_ROW_POOL_H
#include <stdlib.h>

typedef int (*cpu_rows_function)(void *context, int start, int stop);

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <xmmintrin.h>

typedef struct cpu_rows_task {
    cpu_rows_function function;
    void *context;
    int start;
    int stop;
    int status;
    unsigned int mxcsr;
} cpu_rows_task;

typedef struct cpu_rows_pool {
    PTP_POOL pool;
    TP_CALLBACK_ENVIRON environment;
    PTP_WORK work[7];
    cpu_rows_task task[7];
    int threads;
    int work_count;
    volatile LONG busy;
} cpu_rows_pool;

static VOID CALLBACK cpu_rows_callback(PTP_CALLBACK_INSTANCE instance, PVOID argument, PTP_WORK work)
{
    cpu_rows_task *task = (cpu_rows_task *)argument;
    unsigned int previous = _mm_getcsr();
    (void)instance;
    (void)work;
    /* Preserve the caller's rounding/denormal mode across worker threads. */
    _mm_setcsr(task->mxcsr);
    task->status = task->function(task->context, task->start, task->stop);
    _mm_setcsr(previous);
}

static void cpu_rows_pool_destroy(cpu_rows_pool *pool)
{
    int index;
    if (pool == NULL) { return; }
    /* Python also serializes close and calls; this protects direct ABI users. */
    while (InterlockedCompareExchange(&pool->busy, 0, 0) != 0) { Sleep(1); }
    for (index = 0; index < pool->work_count; ++index) {
        WaitForThreadpoolWorkCallbacks(pool->work[index], FALSE);
        CloseThreadpoolWork(pool->work[index]);
    }
    if (pool->pool != NULL) {
        DestroyThreadpoolEnvironment(&pool->environment);
        CloseThreadpool(pool->pool);
    }
    free(pool);
}

static cpu_rows_pool *cpu_rows_pool_create(int threads)
{
    cpu_rows_pool *pool;
    int index;
    if (threads < 1 || threads > 8) { return NULL; }
    pool = (cpu_rows_pool *)calloc(1, sizeof(*pool));
    if (pool == NULL) { return NULL; }
    pool->threads = threads;
    if (threads == 1) { return pool; }
    pool->pool = CreateThreadpool(NULL);
    if (pool->pool == NULL) { free(pool); return NULL; }
    InitializeThreadpoolEnvironment(&pool->environment);
    SetThreadpoolCallbackPool(&pool->environment, pool->pool);
    SetThreadpoolThreadMaximum(pool->pool, (DWORD)(threads - 1));
    if (!SetThreadpoolThreadMinimum(pool->pool, (DWORD)(threads - 1))) {
        cpu_rows_pool_destroy(pool);
        return NULL;
    }
    for (index = 0; index < threads - 1; ++index) {
        pool->work[index] = CreateThreadpoolWork(cpu_rows_callback, &pool->task[index], &pool->environment);
        if (pool->work[index] == NULL) { cpu_rows_pool_destroy(pool); return NULL; }
        pool->work_count += 1;
    }
    return pool;
}

static int cpu_rows_run(cpu_rows_pool *pool, cpu_rows_function function, void *context, int rows, int parallelize)
{
    int count, index, status;
    if (pool == NULL || pool->threads == 1 || rows == 1 || !parallelize) { return function(context, 0, rows); }
    if (InterlockedCompareExchange(&pool->busy, 1, 0) != 0) { return 3; }
    count = rows < pool->threads ? rows : pool->threads;
    for (index = 1; index < count; ++index) {
        cpu_rows_task *task = &pool->task[index - 1];
        task->function = function;
        task->context = context;
        task->start = rows * index / count;
        task->stop = rows * (index + 1) / count;
        task->status = 0;
        task->mxcsr = _mm_getcsr();
        SubmitThreadpoolWork(pool->work[index - 1]);
    }
    status = function(context, 0, rows / count);
    for (index = 1; index < count; ++index) {
        WaitForThreadpoolWorkCallbacks(pool->work[index - 1], FALSE);
        if (status == 0) { status = pool->task[index - 1].status; }
    }
    InterlockedExchange(&pool->busy, 0);
    return status;
}
#else
typedef struct cpu_rows_pool { int threads; } cpu_rows_pool;
static cpu_rows_pool *cpu_rows_pool_create(int threads)
{
    cpu_rows_pool *pool;
    if (threads != 1) { return NULL; }
    pool = (cpu_rows_pool *)malloc(sizeof(*pool));
    if (pool != NULL) { pool->threads = 1; }
    return pool;
}
static void cpu_rows_pool_destroy(cpu_rows_pool *pool) { free(pool); }
static int cpu_rows_run(cpu_rows_pool *pool, cpu_rows_function function, void *context, int rows, int parallelize)
{
    (void)pool;
    (void)parallelize;
    return function(context, 0, rows);
}
#endif
#endif
