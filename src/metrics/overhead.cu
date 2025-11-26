/*
 * GPU Virtualization Performance Evaluation Tool
 * Overhead Metrics Benchmarks (OH-001 to OH-010)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "include/benchmark.h"
#include "include/metrics.h"

/*
 * ============================================================================
 * CUDA Kernels for Overhead Measurement
 * ============================================================================
 */

/* Empty kernel - measures pure launch overhead */
__global__ void kernel_empty(void) {
    /* Intentionally empty */
}

/* Minimal compute kernel - single operation */
__global__ void kernel_minimal(int *out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *out = 1;
    }
}

/* Variable size kernel - tests rate limiter with different grid/block sizes */
__global__ void kernel_variable_size(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}

/* Compute-bound kernel for SM utilization testing */
__global__ void kernel_compute_bound(float *data, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (float)idx;

    for (int i = 0; i < iterations; i++) {
        val = sinf(val) * cosf(val) + sqrtf(fabsf(val));
    }

    if (data != NULL) {
        data[idx % 1024] = val;
    }
}

/*
 * ============================================================================
 * OH-001: Kernel Launch Latency
 * Measures time from cuLaunchKernel call to kernel execution start
 * ============================================================================
 */

int bench_kernel_launch_latency(metric_result_t *result) {
    const int iterations = 1000;
    const int warmup = 100;

    strcpy(result->metric_id, METRIC_OH_KERNEL_LAUNCH_LATENCY);
    result->device_id = 0;
    result->valid = false;

    /* Allocate result storage */
    result->raw_values = (double*)malloc(sizeof(double) * iterations);
    if (result->raw_values == NULL) {
        strcpy(result->error_msg, "Failed to allocate result storage");
        return -1;
    }
    result->raw_count = 0;

    cudaStream_t stream;
    cudaError_t err = cudaStreamCreate(&stream);
    if (err != cudaSuccess) {
        sprintf(result->error_msg, "cudaStreamCreate failed: %s", cudaGetErrorString(err));
        return -1;
    }

    /* Warmup phase */
    for (int i = 0; i < warmup; i++) {
        kernel_empty<<<1, 1, 0, stream>>>();
    }
    cudaStreamSynchronize(stream);

    /* Measurement phase */
    for (int i = 0; i < iterations; i++) {
        timing_result_t t;
        timing_start(&t);

        kernel_empty<<<1, 1, 0, stream>>>();
        cudaStreamSynchronize(stream);

        timing_stop(&t);
        result->raw_values[i] = t.elapsed_us;
    }
    result->raw_count = iterations;

    /* Calculate statistics */
    stats_calculate(result->raw_values, iterations, &result->stats);
    result->value = result->stats.median;  /* Use median as primary value */
    result->timestamp_ns = timing_get_ns();
    result->valid = true;

    cudaStreamDestroy(stream);

    LOG_INFO("OH-001 Kernel Launch Latency: %.2f us (median), %.2f us (mean)",
             result->stats.median, result->stats.mean);

    return 0;
}

/*
 * ============================================================================
 * OH-002: Memory Allocation Latency
 * Measures cuMemAlloc time including virtualization overhead
 * ============================================================================
 */

int bench_memory_allocation_latency(metric_result_t *result) {
    const size_t test_sizes[] = {
        1024,           /* 1 KB */
        65536,          /* 64 KB */
        1048576,        /* 1 MB */
        16777216,       /* 16 MB */
        67108864,       /* 64 MB */
        268435456,      /* 256 MB */
    };
    const int num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);
    const int iterations_per_size = 100;
    const int warmup = 10;

    strcpy(result->metric_id, METRIC_OH_MEMORY_ALLOC_LATENCY);
    result->device_id = 0;
    result->valid = false;

    int total_iterations = num_sizes * iterations_per_size;
    result->raw_values = (double*)malloc(sizeof(double) * total_iterations);
    if (result->raw_values == NULL) {
        strcpy(result->error_msg, "Failed to allocate result storage");
        return -1;
    }
    result->raw_count = 0;

    /* Warmup */
    for (int i = 0; i < warmup; i++) {
        void *ptr;
        cudaMalloc(&ptr, 1048576);
        cudaFree(ptr);
    }

    /* Measurement for each size */
    for (int s = 0; s < num_sizes; s++) {
        size_t size = test_sizes[s];

        for (int i = 0; i < iterations_per_size; i++) {
            void *ptr;
            timing_result_t t;

            timing_start(&t);
            cudaError_t err = cudaMalloc(&ptr, size);
            timing_stop(&t);

            if (err == cudaSuccess) {
                result->raw_values[result->raw_count++] = t.elapsed_us;
                cudaFree(ptr);
            }
        }
    }

    /* Calculate statistics */
    stats_calculate(result->raw_values, result->raw_count, &result->stats);
    result->value = result->stats.median;
    result->timestamp_ns = timing_get_ns();
    result->valid = true;

    LOG_INFO("OH-002 Memory Allocation Latency: %.2f us (median), %.2f us (mean)",
             result->stats.median, result->stats.mean);

    return 0;
}

/*
 * ============================================================================
 * OH-003: Memory Free Latency
 * ============================================================================
 */

int bench_memory_free_latency(metric_result_t *result) {
    const size_t test_sizes[] = {
        1048576,        /* 1 MB */
        16777216,       /* 16 MB */
        67108864,       /* 64 MB */
    };
    const int num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);
    const int iterations_per_size = 100;

    strcpy(result->metric_id, METRIC_OH_MEMORY_FREE_LATENCY);
    result->device_id = 0;
    result->valid = false;

    int total_iterations = num_sizes * iterations_per_size;
    result->raw_values = (double*)malloc(sizeof(double) * total_iterations);
    if (result->raw_values == NULL) {
        strcpy(result->error_msg, "Failed to allocate result storage");
        return -1;
    }
    result->raw_count = 0;

    for (int s = 0; s < num_sizes; s++) {
        size_t size = test_sizes[s];

        for (int i = 0; i < iterations_per_size; i++) {
            void *ptr;
            cudaError_t err = cudaMalloc(&ptr, size);
            if (err != cudaSuccess) continue;

            timing_result_t t;
            timing_start(&t);
            cudaFree(ptr);
            timing_stop(&t);

            result->raw_values[result->raw_count++] = t.elapsed_us;
        }
    }

    stats_calculate(result->raw_values, result->raw_count, &result->stats);
    result->value = result->stats.median;
    result->timestamp_ns = timing_get_ns();
    result->valid = true;

    LOG_INFO("OH-003 Memory Free Latency: %.2f us (median)", result->stats.median);

    return 0;
}

/*
 * ============================================================================
 * OH-004: Context Creation Overhead
 * ============================================================================
 */

int bench_context_creation_overhead(metric_result_t *result) {
    const int iterations = 20;  /* Context creation is expensive, fewer iterations */

    strcpy(result->metric_id, METRIC_OH_CONTEXT_CREATION);
    result->device_id = 0;
    result->valid = false;

    result->raw_values = (double*)malloc(sizeof(double) * iterations);
    if (result->raw_values == NULL) {
        strcpy(result->error_msg, "Failed to allocate result storage");
        return -1;
    }
    result->raw_count = 0;

    /* Reset device to ensure fresh context each time */
    cudaDeviceReset();

    for (int i = 0; i < iterations; i++) {
        timing_result_t t;

        /* Measure context creation via cudaSetDevice + first operation */
        timing_start(&t);

        cudaSetDevice(0);

        /* Force context creation with a minimal operation */
        void *ptr;
        cudaMalloc(&ptr, 1024);
        cudaFree(ptr);

        timing_stop(&t);

        result->raw_values[result->raw_count++] = t.elapsed_us;

        /* Reset for next iteration */
        cudaDeviceReset();
    }

    stats_calculate(result->raw_values, result->raw_count, &result->stats);
    result->value = result->stats.median;
    result->timestamp_ns = timing_get_ns();
    result->valid = true;

    LOG_INFO("OH-004 Context Creation Overhead: %.2f us (median)", result->stats.median);

    return 0;
}

/*
 * ============================================================================
 * OH-005: API Interception Overhead
 * Measures dlsym hook overhead using simple CUDA calls
 * ============================================================================
 */

int bench_api_interception_overhead(metric_result_t *result) {
    const int iterations = 10000;
    const int warmup = 1000;

    strcpy(result->metric_id, METRIC_OH_API_INTERCEPTION);
    result->device_id = 0;
    result->valid = false;

    result->raw_values = (double*)malloc(sizeof(double) * iterations);
    if (result->raw_values == NULL) {
        strcpy(result->error_msg, "Failed to allocate result storage");
        return -1;
    }
    result->raw_count = 0;

    int device_count;

    /* Warmup */
    for (int i = 0; i < warmup; i++) {
        cudaGetDeviceCount(&device_count);
    }

    /* Measurement - use a simple API call that doesn't do much work */
    for (int i = 0; i < iterations; i++) {
        timing_result_t t;

        timing_start(&t);
        cudaGetDeviceCount(&device_count);
        timing_stop(&t);

        /* Convert to nanoseconds */
        result->raw_values[result->raw_count++] = t.elapsed_us * 1000.0;
    }

    stats_calculate(result->raw_values, result->raw_count, &result->stats);
    result->value = result->stats.median;  /* In nanoseconds */
    result->timestamp_ns = timing_get_ns();
    result->valid = true;

    LOG_INFO("OH-005 API Interception Overhead: %.0f ns (median)", result->stats.median);

    return 0;
}

/*
 * ============================================================================
 * OH-006: Shared Region Lock Contention
 * Measures lock wait time under multi-threaded contention
 * ============================================================================
 */

typedef struct {
    int thread_id;
    int iterations;
    double *latencies;
    int latency_count;
} contention_thread_args_t;

static void* contention_thread_func(void *arg) {
    contention_thread_args_t *args = (contention_thread_args_t*)arg;

    /* Each thread allocates and frees memory repeatedly */
    for (int i = 0; i < args->iterations; i++) {
        void *ptr;
        timing_result_t t;

        timing_start(&t);
        cudaError_t err = cudaMalloc(&ptr, 1048576);  /* 1 MB */
        timing_stop(&t);

        if (err == cudaSuccess) {
            args->latencies[args->latency_count++] = t.elapsed_us;
            cudaFree(ptr);
        }
    }

    return NULL;
}

int bench_lock_contention(metric_result_t *result) {
    const int num_threads = 8;
    const int iterations_per_thread = 50;

    strcpy(result->metric_id, METRIC_OH_LOCK_CONTENTION);
    result->device_id = 0;
    result->valid = false;

    int total_iterations = num_threads * iterations_per_thread;
    result->raw_values = (double*)malloc(sizeof(double) * total_iterations);
    if (result->raw_values == NULL) {
        strcpy(result->error_msg, "Failed to allocate result storage");
        return -1;
    }
    result->raw_count = 0;

    pthread_t threads[num_threads];
    contention_thread_args_t thread_args[num_threads];

    /* Initialize thread arguments */
    for (int i = 0; i < num_threads; i++) {
        thread_args[i].thread_id = i;
        thread_args[i].iterations = iterations_per_thread;
        thread_args[i].latencies = (double*)malloc(sizeof(double) * iterations_per_thread);
        thread_args[i].latency_count = 0;
    }

    /* Start threads */
    for (int i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], NULL, contention_thread_func, &thread_args[i]);
    }

    /* Wait for completion */
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    /* Collect results */
    for (int i = 0; i < num_threads; i++) {
        for (int j = 0; j < thread_args[i].latency_count; j++) {
            result->raw_values[result->raw_count++] = thread_args[i].latencies[j];
        }
        free(thread_args[i].latencies);
    }

    stats_calculate(result->raw_values, result->raw_count, &result->stats);
    result->value = result->stats.p95;  /* Use P95 to capture contention impact */
    result->timestamp_ns = timing_get_ns();
    result->valid = true;

    LOG_INFO("OH-006 Lock Contention (P95): %.2f us", result->stats.p95);

    return 0;
}

/*
 * ============================================================================
 * OH-007: Memory Tracking Overhead
 * Isolates memory tracking cost from lock contention
 * ============================================================================
 */

int bench_memory_tracking_overhead(metric_result_t *result) {
    /* This benchmark compares allocation time with minimal vs. many tracked allocations */
    const int iterations = 100;
    const size_t alloc_size = 1048576;  /* 1 MB */

    strcpy(result->metric_id, METRIC_OH_MEMORY_TRACKING);
    result->device_id = 0;
    result->valid = false;

    result->raw_values = (double*)malloc(sizeof(double) * iterations);
    if (result->raw_values == NULL) {
        strcpy(result->error_msg, "Failed to allocate result storage");
        return -1;
    }
    result->raw_count = 0;

    /* First, measure baseline with no other allocations */
    cudaDeviceReset();
    double baseline_sum = 0.0;
    for (int i = 0; i < iterations; i++) {
        void *ptr;
        timing_result_t t;
        timing_start(&t);
        cudaMalloc(&ptr, alloc_size);
        timing_stop(&t);
        baseline_sum += t.elapsed_us;
        cudaFree(ptr);
    }
    double baseline_mean = baseline_sum / iterations;

    /* Now allocate many buffers to increase tracking list size */
    void *tracked_ptrs[100];
    for (int i = 0; i < 100; i++) {
        cudaMalloc(&tracked_ptrs[i], alloc_size);
    }

    /* Measure allocation time with many tracked allocations */
    for (int i = 0; i < iterations; i++) {
        void *ptr;
        timing_result_t t;
        timing_start(&t);
        cudaMalloc(&ptr, alloc_size);
        timing_stop(&t);

        /* Store overhead relative to baseline in nanoseconds */
        double overhead_ns = (t.elapsed_us - baseline_mean) * 1000.0;
        result->raw_values[result->raw_count++] = overhead_ns > 0 ? overhead_ns : 0;
        cudaFree(ptr);
    }

    /* Cleanup tracked allocations */
    for (int i = 0; i < 100; i++) {
        cudaFree(tracked_ptrs[i]);
    }

    stats_calculate(result->raw_values, result->raw_count, &result->stats);
    result->value = result->stats.median;
    result->timestamp_ns = timing_get_ns();
    result->valid = true;

    LOG_INFO("OH-007 Memory Tracking Overhead: %.0f ns (median)", result->stats.median);

    return 0;
}

/*
 * ============================================================================
 * OH-008: Rate Limiter Overhead
 * Measures token bucket check latency
 * ============================================================================
 */

int bench_rate_limiter_overhead(metric_result_t *result) {
    const int iterations = 1000;

    strcpy(result->metric_id, METRIC_OH_RATE_LIMITER);
    result->device_id = 0;
    result->valid = false;

    result->raw_values = (double*)malloc(sizeof(double) * iterations);
    if (result->raw_values == NULL) {
        strcpy(result->error_msg, "Failed to allocate result storage");
        return -1;
    }
    result->raw_count = 0;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    float *d_data;
    cudaMalloc(&d_data, sizeof(float) * 1024);

    /* Test with various kernel sizes to exercise rate limiter */
    int grid_sizes[] = {1, 32, 128, 512};
    int block_sizes[] = {32, 128, 256, 512};

    for (int g = 0; g < 4; g++) {
        for (int b = 0; b < 4; b++) {
            int grid = grid_sizes[g];
            int block = block_sizes[b];
            int n = grid * block;

            for (int i = 0; i < iterations / 16; i++) {
                timing_result_t t;

                timing_start(&t);
                kernel_variable_size<<<grid, block, 0, stream>>>(d_data, n);
                /* Note: We don't sync here to isolate launch overhead */
                timing_stop(&t);

                result->raw_values[result->raw_count++] = t.elapsed_us * 1000.0;  /* to ns */
            }
            cudaStreamSynchronize(stream);
        }
    }

    cudaFree(d_data);
    cudaStreamDestroy(stream);

    stats_calculate(result->raw_values, result->raw_count, &result->stats);
    result->value = result->stats.median;
    result->timestamp_ns = timing_get_ns();
    result->valid = true;

    LOG_INFO("OH-008 Rate Limiter Overhead: %.0f ns (median)", result->stats.median);

    return 0;
}

/*
 * ============================================================================
 * OH-009: NVML Polling Overhead
 * Measures CPU utilization of monitoring thread with actual NVML polling
 * ============================================================================
 */

#include <sys/resource.h>
#include <dlfcn.h>

/* NVML types for dynamic loading */
typedef int nvmlReturn_t;
typedef void* nvmlDevice_t;
typedef struct {
    unsigned int gpu;
    unsigned int memory;
} nvmlUtilization_t;

typedef nvmlReturn_t (*nvmlInit_t)(void);
typedef nvmlReturn_t (*nvmlShutdown_t)(void);
typedef nvmlReturn_t (*nvmlDeviceGetHandleByIndex_t)(unsigned int, nvmlDevice_t*);
typedef nvmlReturn_t (*nvmlDeviceGetUtilizationRates_t)(nvmlDevice_t, nvmlUtilization_t*);
typedef nvmlReturn_t (*nvmlDeviceGetMemoryInfo_t)(nvmlDevice_t, void*);

int bench_nvml_polling_overhead(metric_result_t *result) {
    strcpy(result->metric_id, METRIC_OH_NVML_POLLING);
    result->device_id = 0;
    result->valid = false;

    /* Measure CPU usage over a period of time with active NVML polling */
    const int duration_ms = 1000;
    const int iterations = 5;
    const int polling_interval_ms = 100;  /* Poll every 100ms like typical monitoring */

    result->raw_values = (double*)malloc(sizeof(double) * iterations);
    if (result->raw_values == NULL) {
        strcpy(result->error_msg, "Failed to allocate result storage");
        return -1;
    }
    result->raw_count = 0;

    /* Try to load NVML */
    void *nvml_lib = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
    if (!nvml_lib) nvml_lib = dlopen("libnvidia-ml.so", RTLD_LAZY);

    nvmlInit_t nvmlInit = NULL;
    nvmlShutdown_t nvmlShutdown = NULL;
    nvmlDeviceGetHandleByIndex_t nvmlDeviceGetHandleByIndex = NULL;
    nvmlDeviceGetUtilizationRates_t nvmlDeviceGetUtilizationRates = NULL;

    bool use_nvml = false;
    nvmlDevice_t device = NULL;

    if (nvml_lib) {
        nvmlInit = (nvmlInit_t)dlsym(nvml_lib, "nvmlInit_v2");
        if (!nvmlInit) nvmlInit = (nvmlInit_t)dlsym(nvml_lib, "nvmlInit");
        nvmlShutdown = (nvmlShutdown_t)dlsym(nvml_lib, "nvmlShutdown");
        nvmlDeviceGetHandleByIndex = (nvmlDeviceGetHandleByIndex_t)dlsym(nvml_lib, "nvmlDeviceGetHandleByIndex_v2");
        if (!nvmlDeviceGetHandleByIndex) nvmlDeviceGetHandleByIndex = (nvmlDeviceGetHandleByIndex_t)dlsym(nvml_lib, "nvmlDeviceGetHandleByIndex");
        nvmlDeviceGetUtilizationRates = (nvmlDeviceGetUtilizationRates_t)dlsym(nvml_lib, "nvmlDeviceGetUtilizationRates");

        if (nvmlInit && nvmlShutdown && nvmlDeviceGetHandleByIndex && nvmlDeviceGetUtilizationRates) {
            if (nvmlInit() == 0) { /* NVML_SUCCESS */
                if (nvmlDeviceGetHandleByIndex(0, &device) == 0) {
                    use_nvml = true;
                }
            }
        }
    }

    for (int i = 0; i < iterations; i++) {
        struct rusage start_usage, end_usage;

        getrusage(RUSAGE_SELF, &start_usage);

        /* Actively poll NVML for the duration (simulating monitoring thread) */
        int polls_remaining = duration_ms / polling_interval_ms;
        while (polls_remaining > 0) {
            if (use_nvml) {
                nvmlUtilization_t util;
                nvmlDeviceGetUtilizationRates(device, &util);
                /* Just reading, don't need the value */
                (void)util;
            }
            usleep(polling_interval_ms * 1000);
            polls_remaining--;
        }

        getrusage(RUSAGE_SELF, &end_usage);

        /* Calculate CPU time used in seconds */
        double user_sec = (end_usage.ru_utime.tv_sec - start_usage.ru_utime.tv_sec) +
                          (end_usage.ru_utime.tv_usec - start_usage.ru_utime.tv_usec) / 1e6;
        double sys_sec = (end_usage.ru_stime.tv_sec - start_usage.ru_stime.tv_sec) +
                         (end_usage.ru_stime.tv_usec - start_usage.ru_stime.tv_usec) / 1e6;

        double total_cpu_sec = user_sec + sys_sec;
        double wall_sec = duration_ms / 1000.0;

        /* CPU utilization percentage */
        double cpu_util = (total_cpu_sec / wall_sec) * 100.0;

        result->raw_values[result->raw_count++] = cpu_util;
    }

    if (use_nvml && nvmlShutdown) {
        nvmlShutdown();
    }
    if (nvml_lib) {
        dlclose(nvml_lib);
    }

    stats_calculate(result->raw_values, result->raw_count, &result->stats);
    result->value = result->stats.mean;
    result->timestamp_ns = timing_get_ns();
    result->valid = true;

    LOG_INFO("OH-009 NVML Polling Overhead: %.2f%% CPU (NVML: %s)",
             result->stats.mean, use_nvml ? "available" : "fallback");

    return 0;
}

/*
 * ============================================================================
 * OH-010: Total Throughput Degradation
 * End-to-end performance measurement vs native baseline
 * ============================================================================
 */

int bench_total_throughput(metric_result_t *result) {
    const int iterations = 100;
    const int kernel_iterations = 100;
    const int grid_size = 256;
    const int block_size = 256;

    strcpy(result->metric_id, METRIC_OH_TOTAL_THROUGHPUT);
    result->device_id = 0;
    result->valid = false;

    result->raw_values = (double*)malloc(sizeof(double) * iterations);
    if (result->raw_values == NULL) {
        strcpy(result->error_msg, "Failed to allocate result storage");
        return -1;
    }
    result->raw_count = 0;

    float *d_data;
    cudaError_t err = cudaMalloc(&d_data, sizeof(float) * grid_size * block_size);
    if (err != cudaSuccess) {
        sprintf(result->error_msg, "cudaMalloc failed: %s", cudaGetErrorString(err));
        return -1;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    /* Warmup */
    for (int i = 0; i < 10; i++) {
        kernel_compute_bound<<<grid_size, block_size, 0, stream>>>(d_data, 100);
    }
    cudaStreamSynchronize(stream);

    /* Measurement: complete workflow including allocate, compute, free */
    for (int i = 0; i < iterations; i++) {
        timing_result_t t;

        timing_start(&t);

        /* Allocate */
        float *d_temp;
        cudaMalloc(&d_temp, sizeof(float) * 1024 * 1024);

        /* Compute */
        for (int k = 0; k < kernel_iterations; k++) {
            kernel_compute_bound<<<grid_size, block_size, 0, stream>>>(d_data, 50);
        }
        cudaStreamSynchronize(stream);

        /* Free */
        cudaFree(d_temp);

        timing_stop(&t);

        result->raw_values[result->raw_count++] = t.elapsed_ms;
    }

    cudaFree(d_data);
    cudaStreamDestroy(stream);

    stats_calculate(result->raw_values, result->raw_count, &result->stats);

    /* Result is throughput relative to baseline (stored as degradation %) */
    /* For comparison, native baseline would need to be loaded from reference */
    result->value = 0.0;  /* Will be calculated during comparison phase */
    result->timestamp_ns = timing_get_ns();
    result->valid = true;

    LOG_INFO("OH-010 Total Throughput: %.2f ms per iteration (mean)", result->stats.mean);

    return 0;
}

/*
 * ============================================================================
 * Run All Overhead Benchmarks
 * ============================================================================
 */

int bench_run_overhead(const benchmark_config_t *config, benchmark_result_t *results) {
    strcpy(results->benchmark_name, "Overhead Benchmarks");

    /* Allocate space for all overhead metrics */
    results->results = (metric_result_t*)calloc(10, sizeof(metric_result_t));
    if (results->results == NULL) {
        strcpy(results->error_msg, "Failed to allocate metric results");
        results->success = false;
        return -1;
    }
    results->result_count = 0;

    timing_result_t total_time;
    timing_start(&total_time);

    /* Run each overhead benchmark */
    LOG_INFO("Running OH-001: Kernel Launch Latency");
    if (bench_kernel_launch_latency(&results->results[results->result_count]) == 0) {
        results->result_count++;
    }

    LOG_INFO("Running OH-002: Memory Allocation Latency");
    if (bench_memory_allocation_latency(&results->results[results->result_count]) == 0) {
        results->result_count++;
    }

    LOG_INFO("Running OH-003: Memory Free Latency");
    if (bench_memory_free_latency(&results->results[results->result_count]) == 0) {
        results->result_count++;
    }

    LOG_INFO("Running OH-004: Context Creation Overhead");
    if (bench_context_creation_overhead(&results->results[results->result_count]) == 0) {
        results->result_count++;
    }

    LOG_INFO("Running OH-005: API Interception Overhead");
    if (bench_api_interception_overhead(&results->results[results->result_count]) == 0) {
        results->result_count++;
    }

    LOG_INFO("Running OH-006: Lock Contention");
    if (bench_lock_contention(&results->results[results->result_count]) == 0) {
        results->result_count++;
    }

    LOG_INFO("Running OH-007: Memory Tracking Overhead");
    if (bench_memory_tracking_overhead(&results->results[results->result_count]) == 0) {
        results->result_count++;
    }

    LOG_INFO("Running OH-008: Rate Limiter Overhead");
    if (bench_rate_limiter_overhead(&results->results[results->result_count]) == 0) {
        results->result_count++;
    }

    LOG_INFO("Running OH-009: NVML Polling Overhead");
    if (bench_nvml_polling_overhead(&results->results[results->result_count]) == 0) {
        results->result_count++;
    }

    LOG_INFO("Running OH-010: Total Throughput");
    if (bench_total_throughput(&results->results[results->result_count]) == 0) {
        results->result_count++;
    }

    timing_stop(&total_time);
    results->total_time_ms = total_time.elapsed_ms;
    results->success = (results->result_count > 0);

    LOG_INFO("Overhead benchmarks completed: %d/%d metrics, %.2f ms total",
             results->result_count, 10, results->total_time_ms);

    return results->success ? 0 : -1;
}
