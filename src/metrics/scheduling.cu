/*
 * GPU Scheduling and Context Switch Metrics (SCHED-001 to SCHED-004)
 *
 * Measures GPU scheduling behavior and context switching overhead
 * for evaluating virtualization frameworks' scheduling efficiency.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <sys/wait.h>
#include <cuda_runtime.h>

extern "C" {
#include "include/benchmark.h"
#include "include/metrics.h"
}

/* Configuration */
#define SCHED_WARMUP_ITERATIONS   5
#define SCHED_TEST_ITERATIONS     50
#define SCHED_CONTEXT_SWITCHES    100

/* Simple compute kernel for scheduling tests */
__global__ void sched_compute_kernel(float *data, size_t n, int iterations) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < n; i += stride) {
        float val = data[i];
        for (int iter = 0; iter < iterations; iter++) {
            val = sinf(val) * cosf(val) + 0.1f;
        }
        data[i] = val;
    }
}

/* Very short kernel for context switch measurement */
__global__ void minimal_kernel(int *flag) {
    if (threadIdx.x == 0) {
        *flag = 1;
    }
}

/* Long-running kernel that can be preempted */
__global__ void long_running_kernel(float *data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < n; i += stride) {
        float val = data[i];
        for (int iter = 0; iter < 1000; iter++) {
            val = __sinf(val) * __cosf(val) + 0.001f;
        }
        data[i] = val;
    }
}

/* Get current time in microseconds */
static double get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1000.0;
}

/*
 * SCHED-001: Context Switch Latency
 *
 * Measures the time to switch between CUDA contexts.
 * Important for multi-tenant GPU sharing.
 */
static bench_result_t bench_context_switch_latency(bench_config_t *config) {
    bench_result_t result;
    memset(&result, 0, sizeof(result));
    strncpy(result.metric_id, METRIC_SCHED_CONTEXT_SWITCH, sizeof(result.metric_id) - 1);
    strncpy(result.name, "Context Switch Latency", sizeof(result.name) - 1);
    strncpy(result.unit, "us", sizeof(result.unit) - 1);

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (err != cudaSuccess || device_count < 1) {
        result.success = 0;
        snprintf(result.error_message, sizeof(result.error_message),
                 "No CUDA devices available");
        return result;
    }

    /* Measure context switch by creating and switching between streams */
    int num_streams = 8;
    cudaStream_t *streams = (cudaStream_t*)malloc(num_streams * sizeof(cudaStream_t));
    int *d_flags = NULL;
    int *h_flags = (int*)malloc(num_streams * sizeof(int));

    err = cudaMalloc(&d_flags, num_streams * sizeof(int));
    if (err != cudaSuccess) {
        free(streams);
        free(h_flags);
        result.success = 0;
        snprintf(result.error_message, sizeof(result.error_message),
                 "Failed to allocate device memory");
        return result;
    }

    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
        h_flags[i] = 0;
    }
    cudaMemset(d_flags, 0, num_streams * sizeof(int));

    /* Warmup */
    for (int w = 0; w < SCHED_WARMUP_ITERATIONS; w++) {
        for (int s = 0; s < num_streams; s++) {
            minimal_kernel<<<1, 32, 0, streams[s]>>>(&d_flags[s]);
        }
        cudaDeviceSynchronize();
    }

    /* Measure context switches */
    double total_switch_time = 0.0;
    double switch_times[SCHED_CONTEXT_SWITCHES];
    int valid_measurements = 0;

    for (int iter = 0; iter < SCHED_CONTEXT_SWITCHES; iter++) {
        /* Launch kernel on stream 0 */
        double t1 = get_time_us();
        minimal_kernel<<<1, 32, 0, streams[0]>>>(&d_flags[0]);
        cudaStreamSynchronize(streams[0]);
        double t2 = get_time_us();

        /* Immediately switch to stream 1 */
        minimal_kernel<<<1, 32, 0, streams[1]>>>(&d_flags[1]);
        cudaStreamSynchronize(streams[1]);
        double t3 = get_time_us();

        /* The context switch overhead is approximately (t3-t2) - (t2-t1) */
        double kernel_time = t2 - t1;
        double switch_plus_kernel = t3 - t2;
        double switch_overhead = switch_plus_kernel - kernel_time;

        if (switch_overhead > 0) {
            switch_times[valid_measurements] = switch_overhead;
            total_switch_time += switch_overhead;
            valid_measurements++;
        }
    }

    if (valid_measurements > 0) {
        result.value = total_switch_time / valid_measurements;

        /* Calculate stddev */
        double variance = 0.0;
        for (int i = 0; i < valid_measurements; i++) {
            double diff = switch_times[i] - result.value;
            variance += diff * diff;
        }
        result.stddev = sqrt(variance / valid_measurements);

        result.success = 1;
        result.iterations = valid_measurements;
        snprintf(result.details, sizeof(result.details),
                 "Context switch latency between streams, measurements=%d, latency=%.2f us",
                 valid_measurements, result.value);
    } else {
        result.success = 0;
        snprintf(result.error_message, sizeof(result.error_message),
                 "Could not measure valid context switch times");
    }

    /* Cleanup */
    for (int i = 0; i < num_streams; i++) {
        cudaStreamDestroy(streams[i]);
    }
    cudaFree(d_flags);
    free(streams);
    free(h_flags);

    return result;
}

/*
 * SCHED-002: Kernel Launch Overhead
 *
 * Measures the overhead of launching kernels, which is important
 * for workloads with many small kernel invocations.
 */
static bench_result_t bench_kernel_launch_overhead(bench_config_t *config) {
    bench_result_t result;
    memset(&result, 0, sizeof(result));
    strncpy(result.metric_id, METRIC_SCHED_KERNEL_LAUNCH, sizeof(result.metric_id) - 1);
    strncpy(result.name, "Kernel Launch Overhead", sizeof(result.name) - 1);
    strncpy(result.unit, "us", sizeof(result.unit) - 1);

    int *d_flag;
    cudaError_t err = cudaMalloc(&d_flag, sizeof(int));
    if (err != cudaSuccess) {
        result.success = 0;
        snprintf(result.error_message, sizeof(result.error_message),
                 "Failed to allocate device memory");
        return result;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /* Warmup */
    for (int i = 0; i < SCHED_WARMUP_ITERATIONS * 10; i++) {
        minimal_kernel<<<1, 32>>>(d_flag);
    }
    cudaDeviceSynchronize();

    /* Measure individual kernel launches */
    double launch_times[SCHED_TEST_ITERATIONS];
    double total_time = 0.0;

    for (int iter = 0; iter < SCHED_TEST_ITERATIONS; iter++) {
        cudaEventRecord(start);
        minimal_kernel<<<1, 32>>>(d_flag);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        launch_times[iter] = ms * 1000.0;  /* Convert to microseconds */
        total_time += launch_times[iter];
    }

    result.value = total_time / SCHED_TEST_ITERATIONS;

    /* Calculate stddev */
    double variance = 0.0;
    for (int i = 0; i < SCHED_TEST_ITERATIONS; i++) {
        double diff = launch_times[i] - result.value;
        variance += diff * diff;
    }
    result.stddev = sqrt(variance / SCHED_TEST_ITERATIONS);

    result.success = 1;
    result.iterations = SCHED_TEST_ITERATIONS;
    snprintf(result.details, sizeof(result.details),
             "Minimal kernel launch overhead, iterations=%d, overhead=%.2f us",
             SCHED_TEST_ITERATIONS, result.value);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_flag);

    return result;
}

/*
 * SCHED-003: Stream Concurrency
 *
 * Measures how well the GPU can handle concurrent stream execution.
 * This reveals scheduler efficiency for parallel workloads.
 */
static bench_result_t bench_stream_concurrency(bench_config_t *config) {
    bench_result_t result;
    memset(&result, 0, sizeof(result));
    strncpy(result.metric_id, METRIC_SCHED_STREAM_CONCURRENCY, sizeof(result.metric_id) - 1);
    strncpy(result.name, "Stream Concurrency Efficiency", sizeof(result.name) - 1);
    strncpy(result.unit, "%", sizeof(result.unit) - 1);

    size_t size = 4 * 1024 * 1024;  /* 4 MB per stream */
    size_t num_elements = size / sizeof(float);
    int num_streams = 4;
    int compute_iterations = 100;

    float **d_data = (float**)malloc(num_streams * sizeof(float*));
    cudaStream_t *streams = (cudaStream_t*)malloc(num_streams * sizeof(cudaStream_t));

    for (int s = 0; s < num_streams; s++) {
        cudaError_t err = cudaMalloc(&d_data[s], size);
        if (err != cudaSuccess) {
            for (int j = 0; j < s; j++) cudaFree(d_data[j]);
            free(d_data);
            free(streams);
            result.success = 0;
            snprintf(result.error_message, sizeof(result.error_message),
                     "Failed to allocate device memory");
            return result;
        }
        cudaStreamCreate(&streams[s]);
        cudaMemset(d_data[s], 0, size);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    /* Warmup */
    for (int w = 0; w < SCHED_WARMUP_ITERATIONS; w++) {
        for (int s = 0; s < num_streams; s++) {
            sched_compute_kernel<<<blocks, threads, 0, streams[s]>>>(d_data[s], num_elements, compute_iterations);
        }
        cudaDeviceSynchronize();
    }

    /* Measure sequential execution time (baseline) */
    cudaEventRecord(start);
    for (int s = 0; s < num_streams; s++) {
        sched_compute_kernel<<<blocks, threads>>>(d_data[s], num_elements, compute_iterations);
        cudaDeviceSynchronize();  /* Force sequential */
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float sequential_ms;
    cudaEventElapsedTime(&sequential_ms, start, stop);

    /* Measure concurrent execution time */
    double concurrent_times[SCHED_TEST_ITERATIONS];
    double total_concurrent = 0.0;

    for (int iter = 0; iter < SCHED_TEST_ITERATIONS; iter++) {
        cudaEventRecord(start);

        /* Launch all streams concurrently */
        for (int s = 0; s < num_streams; s++) {
            sched_compute_kernel<<<blocks, threads, 0, streams[s]>>>(d_data[s], num_elements, compute_iterations);
        }

        /* Wait for all to complete */
        for (int s = 0; s < num_streams; s++) {
            cudaStreamSynchronize(streams[s]);
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float concurrent_ms;
        cudaEventElapsedTime(&concurrent_ms, start, stop);
        concurrent_times[iter] = concurrent_ms;
        total_concurrent += concurrent_ms;
    }

    double avg_concurrent = total_concurrent / SCHED_TEST_ITERATIONS;

    /* Calculate concurrency efficiency */
    /* Perfect concurrency: concurrent_time = sequential_time / num_streams */
    /* Efficiency = (expected_concurrent / actual_concurrent) * 100 */
    double expected_concurrent = sequential_ms / num_streams;
    double efficiency = (expected_concurrent / avg_concurrent) * 100.0;

    /* Cap at 100% */
    if (efficiency > 100.0) efficiency = 100.0;

    result.value = efficiency;

    /* Calculate stddev based on time variance */
    double variance = 0.0;
    for (int i = 0; i < SCHED_TEST_ITERATIONS; i++) {
        double diff = concurrent_times[i] - avg_concurrent;
        variance += diff * diff;
    }
    double time_stddev = sqrt(variance / SCHED_TEST_ITERATIONS);
    result.stddev = efficiency * (time_stddev / avg_concurrent);

    result.success = 1;
    result.iterations = SCHED_TEST_ITERATIONS;
    snprintf(result.details, sizeof(result.details),
             "Streams=%d, sequential=%.2f ms, concurrent=%.2f ms, efficiency=%.1f%%",
             num_streams, sequential_ms, avg_concurrent, efficiency);

    /* Cleanup */
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    for (int s = 0; s < num_streams; s++) {
        cudaFree(d_data[s]);
        cudaStreamDestroy(streams[s]);
    }
    free(d_data);
    free(streams);

    return result;
}

/*
 * SCHED-004: Preemption Latency
 *
 * Measures the latency of GPU preemption when higher priority work arrives.
 * Critical for real-time workloads sharing the GPU.
 */
static bench_result_t bench_preemption_latency(bench_config_t *config) {
    bench_result_t result;
    memset(&result, 0, sizeof(result));
    strncpy(result.metric_id, METRIC_SCHED_PREEMPTION, sizeof(result.metric_id) - 1);
    strncpy(result.name, "Preemption Latency", sizeof(result.name) - 1);
    strncpy(result.unit, "ms", sizeof(result.unit) - 1);

    size_t size = 64 * 1024 * 1024;  /* 64 MB */
    size_t num_elements = size / sizeof(float);

    float *d_background, *d_foreground;
    cudaError_t err = cudaMalloc(&d_background, size);
    if (err != cudaSuccess) {
        result.success = 0;
        snprintf(result.error_message, sizeof(result.error_message),
                 "Failed to allocate background memory");
        return result;
    }

    err = cudaMalloc(&d_foreground, size);
    if (err != cudaSuccess) {
        cudaFree(d_background);
        result.success = 0;
        snprintf(result.error_message, sizeof(result.error_message),
                 "Failed to allocate foreground memory");
        return result;
    }

    cudaMemset(d_background, 0, size);
    cudaMemset(d_foreground, 0, size);

    /* Create streams with different priorities */
    int least_priority, greatest_priority;
    cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);

    cudaStream_t low_priority_stream, high_priority_stream;
    cudaStreamCreateWithPriority(&low_priority_stream, cudaStreamNonBlocking, least_priority);
    cudaStreamCreateWithPriority(&high_priority_stream, cudaStreamNonBlocking, greatest_priority);

    cudaEvent_t start, stop, high_start, high_stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&high_start);
    cudaEventCreate(&high_stop);

    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    /* Warmup */
    for (int w = 0; w < SCHED_WARMUP_ITERATIONS; w++) {
        long_running_kernel<<<blocks, threads, 0, low_priority_stream>>>(d_background, num_elements);
        cudaStreamSynchronize(low_priority_stream);
    }

    /* Measure preemption behavior */
    double preemption_times[SCHED_TEST_ITERATIONS];
    double total_preemption = 0.0;
    int valid_measurements = 0;

    for (int iter = 0; iter < SCHED_TEST_ITERATIONS; iter++) {
        /* Launch long-running background kernel */
        cudaEventRecord(start, low_priority_stream);
        long_running_kernel<<<blocks, threads, 0, low_priority_stream>>>(d_background, num_elements);

        /* Wait a bit, then launch high priority kernel */
        usleep(100);  /* 100 microseconds */

        cudaEventRecord(high_start, high_priority_stream);
        sched_compute_kernel<<<blocks/4, threads, 0, high_priority_stream>>>(d_foreground, num_elements/4, 10);
        cudaEventRecord(high_stop, high_priority_stream);

        cudaStreamSynchronize(high_priority_stream);
        cudaEventRecord(stop, low_priority_stream);
        cudaStreamSynchronize(low_priority_stream);

        /* Measure high priority kernel completion time */
        float high_ms;
        cudaEventElapsedTime(&high_ms, high_start, high_stop);

        /* Baseline: measure the high priority kernel alone */
        cudaEventRecord(high_start, high_priority_stream);
        sched_compute_kernel<<<blocks/4, threads, 0, high_priority_stream>>>(d_foreground, num_elements/4, 10);
        cudaEventRecord(high_stop, high_priority_stream);
        cudaStreamSynchronize(high_priority_stream);

        float baseline_ms;
        cudaEventElapsedTime(&baseline_ms, high_start, high_stop);

        /* Preemption latency is the additional time when competing */
        double preemption_overhead = high_ms - baseline_ms;
        if (preemption_overhead > 0) {
            preemption_times[valid_measurements] = preemption_overhead;
            total_preemption += preemption_overhead;
            valid_measurements++;
        }
    }

    if (valid_measurements > 0) {
        result.value = total_preemption / valid_measurements;

        /* Calculate stddev */
        double variance = 0.0;
        for (int i = 0; i < valid_measurements; i++) {
            double diff = preemption_times[i] - result.value;
            variance += diff * diff;
        }
        result.stddev = sqrt(variance / valid_measurements);

        result.success = 1;
        result.iterations = valid_measurements;
        snprintf(result.details, sizeof(result.details),
                 "Priority preemption overhead, measurements=%d, latency=%.3f ms",
                 valid_measurements, result.value);
    } else {
        /* No preemption overhead detected - GPU handles priorities well */
        result.value = 0.0;
        result.stddev = 0.0;
        result.success = 1;
        result.iterations = SCHED_TEST_ITERATIONS;
        snprintf(result.details, sizeof(result.details),
                 "No measurable preemption overhead detected (good scheduling)");
    }

    /* Cleanup */
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(high_start);
    cudaEventDestroy(high_stop);
    cudaStreamDestroy(low_priority_stream);
    cudaStreamDestroy(high_priority_stream);
    cudaFree(d_background);
    cudaFree(d_foreground);

    return result;
}

/*
 * Run all scheduling metrics
 */
extern "C" void bench_run_scheduling(bench_config_t *config, bench_result_t *results, int *count) {
    int idx = 0;

    printf("\n=== GPU Scheduling Metrics ===\n\n");

    /* SCHED-001: Context Switch Latency */
    printf("Running SCHED-001: Context Switch Latency...\n");
    results[idx] = bench_context_switch_latency(config);
    if (results[idx].success) {
        printf("  Result: %.2f %s (+/- %.2f)\n",
               results[idx].value, results[idx].unit, results[idx].stddev);
        printf("  %s\n", results[idx].details);
    } else {
        printf("  FAILED: %s\n", results[idx].error_message);
    }
    idx++;

    /* SCHED-002: Kernel Launch Overhead */
    printf("Running SCHED-002: Kernel Launch Overhead...\n");
    results[idx] = bench_kernel_launch_overhead(config);
    if (results[idx].success) {
        printf("  Result: %.2f %s (+/- %.2f)\n",
               results[idx].value, results[idx].unit, results[idx].stddev);
        printf("  %s\n", results[idx].details);
    } else {
        printf("  FAILED: %s\n", results[idx].error_message);
    }
    idx++;

    /* SCHED-003: Stream Concurrency */
    printf("Running SCHED-003: Stream Concurrency Efficiency...\n");
    results[idx] = bench_stream_concurrency(config);
    if (results[idx].success) {
        printf("  Result: %.2f %s (+/- %.2f)\n",
               results[idx].value, results[idx].unit, results[idx].stddev);
        printf("  %s\n", results[idx].details);
    } else {
        printf("  FAILED: %s\n", results[idx].error_message);
    }
    idx++;

    /* SCHED-004: Preemption Latency */
    printf("Running SCHED-004: Preemption Latency...\n");
    results[idx] = bench_preemption_latency(config);
    if (results[idx].success) {
        printf("  Result: %.3f %s (+/- %.3f)\n",
               results[idx].value, results[idx].unit, results[idx].stddev);
        printf("  %s\n", results[idx].details);
    } else {
        printf("  FAILED: %s\n", results[idx].error_message);
    }
    idx++;

    *count = idx;
    printf("\nScheduling metrics completed: %d tests\n", idx);
}
