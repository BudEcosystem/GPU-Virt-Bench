/*
 * GPU Virtualization Performance Evaluation Tool
 * High-precision timing utilities
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#include "include/benchmark.h"

/*
 * ============================================================================
 * CPU Timing (nanosecond precision using clock_gettime)
 * ============================================================================
 */

uint64_t timing_get_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

void timing_start(timing_result_t *t) {
    t->start_ns = timing_get_ns();
    t->end_ns = 0;
    t->elapsed_us = 0.0;
    t->elapsed_ms = 0.0;
}

void timing_stop(timing_result_t *t) {
    t->end_ns = timing_get_ns();
    uint64_t elapsed = t->end_ns - t->start_ns;
    t->elapsed_us = (double)elapsed / 1000.0;
    t->elapsed_ms = (double)elapsed / 1000000.0;
}

/*
 * ============================================================================
 * GPU Timing (using CUDA events for accurate GPU-side measurement)
 * ============================================================================
 */

typedef struct {
    cudaEvent_t start;
    cudaEvent_t stop;
    bool events_created;
} cuda_timer_t;

static __thread cuda_timer_t g_cuda_timer = {0};

static int ensure_cuda_timer_initialized(void) {
    if (!g_cuda_timer.events_created) {
        cudaError_t err;
        err = cudaEventCreate(&g_cuda_timer.start);
        if (err != cudaSuccess) {
            LOG_ERROR("Failed to create CUDA start event: %s", cudaGetErrorString(err));
            return -1;
        }
        err = cudaEventCreate(&g_cuda_timer.stop);
        if (err != cudaSuccess) {
            LOG_ERROR("Failed to create CUDA stop event: %s", cudaGetErrorString(err));
            cudaEventDestroy(g_cuda_timer.start);
            return -1;
        }
        g_cuda_timer.events_created = true;
    }
    return 0;
}

void timing_cuda_sync_start(timing_result_t *t, cudaStream_t stream) {
    if (ensure_cuda_timer_initialized() != 0) {
        t->start_ns = 0;
        return;
    }
    cudaEventRecord(g_cuda_timer.start, stream);
    t->start_ns = timing_get_ns();
}

void timing_cuda_sync_stop(timing_result_t *t, cudaStream_t stream) {
    cudaEventRecord(g_cuda_timer.stop, stream);
    cudaEventSynchronize(g_cuda_timer.stop);
    t->end_ns = timing_get_ns();

    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, g_cuda_timer.start, g_cuda_timer.stop);

    /* Use GPU-measured time for accuracy */
    t->elapsed_ms = (double)gpu_ms;
    t->elapsed_us = (double)gpu_ms * 1000.0;
}

/*
 * ============================================================================
 * Kernel Launch Latency Measurement
 * Measures time from host launch call to kernel execution start
 * ============================================================================
 */

/* Empty kernel for measuring launch overhead */
__global__ void empty_kernel(void) {
    /* Intentionally empty */
}

/* Kernel that records timestamp at start */
__global__ void timestamp_kernel(unsigned long long *d_timestamp) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *d_timestamp = clock64();
    }
}

double measure_kernel_launch_overhead(int iterations) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    double total_us = 0.0;

    /* Warmup */
    for (int i = 0; i < 10; i++) {
        empty_kernel<<<1, 1, 0, stream>>>();
    }
    cudaStreamSynchronize(stream);

    /* Measurement */
    for (int i = 0; i < iterations; i++) {
        timing_result_t t;
        timing_start(&t);
        empty_kernel<<<1, 1, 0, stream>>>();
        cudaStreamSynchronize(stream);
        timing_stop(&t);
        total_us += t.elapsed_us;
    }

    cudaStreamDestroy(stream);
    return total_us / iterations;
}

/*
 * ============================================================================
 * Memory Allocation Timing
 * ============================================================================
 */

double measure_malloc_latency(size_t size, int iterations) {
    double total_us = 0.0;
    void *ptr;

    /* Warmup */
    for (int i = 0; i < 5; i++) {
        cudaMalloc(&ptr, size);
        cudaFree(ptr);
    }

    /* Measurement */
    for (int i = 0; i < iterations; i++) {
        timing_result_t t;
        timing_start(&t);
        cudaError_t err = cudaMalloc(&ptr, size);
        timing_stop(&t);

        if (err == cudaSuccess) {
            total_us += t.elapsed_us;
            cudaFree(ptr);
        }
    }

    return total_us / iterations;
}

double measure_free_latency(size_t size, int iterations) {
    double total_us = 0.0;

    for (int i = 0; i < iterations; i++) {
        void *ptr;
        cudaMalloc(&ptr, size);

        timing_result_t t;
        timing_start(&t);
        cudaFree(ptr);
        timing_stop(&t);

        total_us += t.elapsed_us;
    }

    return total_us / iterations;
}

/*
 * ============================================================================
 * High-Resolution Batch Timing
 * For collecting large numbers of samples efficiently
 * ============================================================================
 */

typedef struct {
    uint64_t *timestamps;
    int capacity;
    int count;
} timestamp_buffer_t;

timestamp_buffer_t* timestamp_buffer_create(int capacity) {
    timestamp_buffer_t *buf = (timestamp_buffer_t*)malloc(sizeof(timestamp_buffer_t));
    if (!buf) return NULL;

    buf->timestamps = (uint64_t*)malloc(sizeof(uint64_t) * capacity * 2);
    if (!buf->timestamps) {
        free(buf);
        return NULL;
    }

    buf->capacity = capacity;
    buf->count = 0;
    return buf;
}

void timestamp_buffer_destroy(timestamp_buffer_t *buf) {
    if (buf) {
        free(buf->timestamps);
        free(buf);
    }
}

void timestamp_buffer_record_start(timestamp_buffer_t *buf) {
    if (buf->count < buf->capacity) {
        buf->timestamps[buf->count * 2] = timing_get_ns();
    }
}

void timestamp_buffer_record_stop(timestamp_buffer_t *buf) {
    if (buf->count < buf->capacity) {
        buf->timestamps[buf->count * 2 + 1] = timing_get_ns();
        buf->count++;
    }
}

void timestamp_buffer_to_results(timestamp_buffer_t *buf, double *results_us) {
    for (int i = 0; i < buf->count; i++) {
        uint64_t start = buf->timestamps[i * 2];
        uint64_t stop = buf->timestamps[i * 2 + 1];
        results_us[i] = (double)(stop - start) / 1000.0;
    }
}

/*
 * ============================================================================
 * RDTSC-based Timing (lowest overhead, platform specific)
 * ============================================================================
 */

#if defined(__x86_64__) || defined(_M_X64)

static inline uint64_t rdtsc(void) {
    unsigned int lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo;
}

static inline uint64_t rdtscp(void) {
    unsigned int lo, hi, aux;
    __asm__ __volatile__ ("rdtscp" : "=a" (lo), "=d" (hi), "=c" (aux));
    return ((uint64_t)hi << 32) | lo;
}

/* Calibrate TSC frequency */
static double g_tsc_freq_mhz = 0.0;

double calibrate_tsc_frequency(void) {
    if (g_tsc_freq_mhz > 0.0) return g_tsc_freq_mhz;

    uint64_t start_tsc = rdtsc();
    uint64_t start_ns = timing_get_ns();

    /* Spin for ~100ms */
    while (timing_get_ns() - start_ns < 100000000ULL) {
        __asm__ __volatile__ ("pause");
    }

    uint64_t end_tsc = rdtsc();
    uint64_t end_ns = timing_get_ns();

    double elapsed_us = (double)(end_ns - start_ns) / 1000.0;
    double tsc_diff = (double)(end_tsc - start_tsc);

    g_tsc_freq_mhz = tsc_diff / elapsed_us;
    return g_tsc_freq_mhz;
}

double tsc_to_us(uint64_t tsc_cycles) {
    if (g_tsc_freq_mhz == 0.0) calibrate_tsc_frequency();
    return (double)tsc_cycles / g_tsc_freq_mhz;
}

#else

double calibrate_tsc_frequency(void) { return 0.0; }
double tsc_to_us(uint64_t tsc_cycles) { return 0.0; }

#endif

/*
 * ============================================================================
 * Cleanup
 * ============================================================================
 */

void timing_cleanup(void) {
    if (g_cuda_timer.events_created) {
        cudaEventDestroy(g_cuda_timer.start);
        cudaEventDestroy(g_cuda_timer.stop);
        g_cuda_timer.events_created = false;
    }
}
