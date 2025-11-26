/*
 * GPU Virtualization Performance Evaluation Tool
 * PCIe Bandwidth Metrics (PCIE-001 to PCIE-004)
 *
 * These metrics measure host-device transfer performance
 * which is a hard shared resource in GPU virtualization.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>
#include <cuda_runtime.h>

extern "C" {
#include "include/benchmark.h"
#include "include/metrics.h"
}

/* Configuration */
#define PCIE_WARMUP_ITERATIONS    3
#define PCIE_TEST_ITERATIONS      20
#define PCIE_NUM_STREAMS          4

/* Get current time in microseconds */
static double get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1000.0;
}

/*
 * Helper Functions
 */

/* Measure H2D bandwidth */
static double measure_h2d_bandwidth(void *h_data, void *d_data, size_t size,
                                   cudaStream_t stream, int iterations) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /* Warmup */
    cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    /* Measure */
    cudaEventRecord(start, stream);
    for (int i = 0; i < iterations; i++) {
        cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    double bytes_transferred = (double)size * (double)iterations;
    double bandwidth_gbps = bytes_transferred / (elapsed_ms / 1000.0) / 1e9;

    return bandwidth_gbps;
}

/* Measure D2H bandwidth */
static double measure_d2h_bandwidth(void *h_data, void *d_data, size_t size,
                                   cudaStream_t stream, int iterations) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /* Warmup */
    cudaMemcpyAsync(h_data, d_data, size, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    /* Measure */
    cudaEventRecord(start, stream);
    for (int i = 0; i < iterations; i++) {
        cudaMemcpyAsync(h_data, d_data, size, cudaMemcpyDeviceToHost, stream);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    double bytes_transferred = (double)size * (double)iterations;
    double bandwidth_gbps = bytes_transferred / (elapsed_ms / 1000.0) / 1e9;

    return bandwidth_gbps;
}

/*
 * PCIE-001: Host-to-Device Bandwidth
 * Measures achieved H2D bandwidth with pinned memory
 */
static bench_result_t bench_pcie_h2d_bandwidth(bench_config_t *config) {
    bench_result_t result;
    memset(&result, 0, sizeof(result));
    strncpy(result.metric_id, METRIC_PCIE_H2D_BANDWIDTH, sizeof(result.metric_id) - 1);
    strncpy(result.name, "H2D Bandwidth", sizeof(result.name) - 1);
    strncpy(result.unit, "GB/s", sizeof(result.unit) - 1);

    const size_t sizes[] = {
        1 * 1024 * 1024,    /* 1 MB */
        16 * 1024 * 1024,   /* 16 MB */
        64 * 1024 * 1024,   /* 64 MB */
        256 * 1024 * 1024,  /* 256 MB */
    };
    const int num_sizes = 4;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    double bandwidths[PCIE_TEST_ITERATIONS];
    double total_bw = 0.0;
    int measurement_idx = 0;

    for (int s = 0; s < num_sizes && measurement_idx < PCIE_TEST_ITERATIONS; s++) {
        size_t size = sizes[s];

        /* Allocate pinned host memory for best performance */
        void *h_data;
        cudaError_t err = cudaHostAlloc(&h_data, size, cudaHostAllocDefault);
        if (err != cudaSuccess) continue;

        void *d_data;
        err = cudaMalloc(&d_data, size);
        if (err != cudaSuccess) {
            cudaFreeHost(h_data);
            continue;
        }

        /* Initialize */
        memset(h_data, 0xAB, size);

        for (int i = 0; i < PCIE_TEST_ITERATIONS / num_sizes && measurement_idx < PCIE_TEST_ITERATIONS; i++) {
            double bw = measure_h2d_bandwidth(h_data, d_data, size, stream, 5);
            bandwidths[measurement_idx] = bw;
            total_bw += bw;
            measurement_idx++;
        }

        cudaFree(d_data);
        cudaFreeHost(h_data);
    }

    cudaStreamDestroy(stream);

    if (measurement_idx == 0) {
        result.success = 0;
        snprintf(result.error_message, sizeof(result.error_message), "No successful measurements");
        return result;
    }

    /* Find max bandwidth */
    double max_bw = 0.0;
    for (int i = 0; i < measurement_idx; i++) {
        if (bandwidths[i] > max_bw) max_bw = bandwidths[i];
    }

    result.value = max_bw;  /* Report peak bandwidth */

    /* Calculate stddev */
    double mean = total_bw / measurement_idx;
    double variance = 0.0;
    for (int i = 0; i < measurement_idx; i++) {
        double diff = bandwidths[i] - mean;
        variance += diff * diff;
    }
    result.stddev = sqrt(variance / measurement_idx);

    result.success = 1;
    result.iterations = measurement_idx;
    snprintf(result.details, sizeof(result.details),
             "Peak H2D bandwidth (pinned memory), sizes 1-256 MB");

    return result;
}

/*
 * PCIE-002: Device-to-Host Bandwidth
 * Measures achieved D2H bandwidth with pinned memory
 */
static bench_result_t bench_pcie_d2h_bandwidth(bench_config_t *config) {
    bench_result_t result;
    memset(&result, 0, sizeof(result));
    strncpy(result.metric_id, METRIC_PCIE_D2H_BANDWIDTH, sizeof(result.metric_id) - 1);
    strncpy(result.name, "D2H Bandwidth", sizeof(result.name) - 1);
    strncpy(result.unit, "GB/s", sizeof(result.unit) - 1);

    const size_t sizes[] = {
        1 * 1024 * 1024,
        16 * 1024 * 1024,
        64 * 1024 * 1024,
        256 * 1024 * 1024,
    };
    const int num_sizes = 4;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    double bandwidths[PCIE_TEST_ITERATIONS];
    double total_bw = 0.0;
    int measurement_idx = 0;

    for (int s = 0; s < num_sizes && measurement_idx < PCIE_TEST_ITERATIONS; s++) {
        size_t size = sizes[s];

        void *h_data;
        cudaError_t err = cudaHostAlloc(&h_data, size, cudaHostAllocDefault);
        if (err != cudaSuccess) continue;

        void *d_data;
        err = cudaMalloc(&d_data, size);
        if (err != cudaSuccess) {
            cudaFreeHost(h_data);
            continue;
        }

        /* Initialize device memory */
        cudaMemset(d_data, 0xCD, size);

        for (int i = 0; i < PCIE_TEST_ITERATIONS / num_sizes && measurement_idx < PCIE_TEST_ITERATIONS; i++) {
            double bw = measure_d2h_bandwidth(h_data, d_data, size, stream, 5);
            bandwidths[measurement_idx] = bw;
            total_bw += bw;
            measurement_idx++;
        }

        cudaFree(d_data);
        cudaFreeHost(h_data);
    }

    cudaStreamDestroy(stream);

    if (measurement_idx == 0) {
        result.success = 0;
        snprintf(result.error_message, sizeof(result.error_message), "No successful measurements");
        return result;
    }

    double max_bw = 0.0;
    for (int i = 0; i < measurement_idx; i++) {
        if (bandwidths[i] > max_bw) max_bw = bandwidths[i];
    }

    result.value = max_bw;

    double mean = total_bw / measurement_idx;
    double variance = 0.0;
    for (int i = 0; i < measurement_idx; i++) {
        double diff = bandwidths[i] - mean;
        variance += diff * diff;
    }
    result.stddev = sqrt(variance / measurement_idx);

    result.success = 1;
    result.iterations = measurement_idx;
    snprintf(result.details, sizeof(result.details),
             "Peak D2H bandwidth (pinned memory), sizes 1-256 MB");

    return result;
}

/*
 * PCIE-003: PCIe Contention Impact
 * Measures bandwidth degradation when multiple streams compete for PCIe
 */

typedef struct {
    void *h_data;
    void *d_data;
    size_t size;
    cudaStream_t stream;
    double bandwidth;
    int iterations;
    int direction;  /* 0 = H2D, 1 = D2H */
} pcie_thread_data_t;

static void* pcie_worker_thread(void *arg) {
    pcie_thread_data_t *data = (pcie_thread_data_t*)arg;

    if (data->direction == 0) {
        data->bandwidth = measure_h2d_bandwidth(data->h_data, data->d_data,
                                               data->size, data->stream, data->iterations);
    } else {
        data->bandwidth = measure_d2h_bandwidth(data->h_data, data->d_data,
                                               data->size, data->stream, data->iterations);
    }

    return NULL;
}

static bench_result_t bench_pcie_contention(bench_config_t *config) {
    bench_result_t result;
    memset(&result, 0, sizeof(result));
    strncpy(result.metric_id, METRIC_PCIE_CONTENTION, sizeof(result.metric_id) - 1);
    strncpy(result.name, "PCIe Contention Impact", sizeof(result.name) - 1);
    strncpy(result.unit, "%", sizeof(result.unit) - 1);

    const int num_streams = PCIE_NUM_STREAMS;  /* Simulate 4 tenants */
    size_t size = 64 * 1024 * 1024;  /* 64 MB per tenant */

    /* Measure baseline (single stream) */
    void *h_baseline;
    void *d_baseline;
    cudaError_t err = cudaHostAlloc(&h_baseline, size, cudaHostAllocDefault);
    if (err != cudaSuccess) {
        result.success = 0;
        snprintf(result.error_message, sizeof(result.error_message),
                 "Failed to allocate baseline host memory");
        return result;
    }

    err = cudaMalloc(&d_baseline, size);
    if (err != cudaSuccess) {
        cudaFreeHost(h_baseline);
        result.success = 0;
        snprintf(result.error_message, sizeof(result.error_message),
                 "Failed to allocate baseline device memory");
        return result;
    }

    memset(h_baseline, 0xAB, size);

    cudaStream_t baseline_stream;
    cudaStreamCreate(&baseline_stream);

    double baseline_bw = measure_h2d_bandwidth(h_baseline, d_baseline, size,
                                               baseline_stream, 10);

    cudaStreamDestroy(baseline_stream);
    cudaFree(d_baseline);
    cudaFreeHost(h_baseline);

    /* Allocate for all "tenants" */
    pcie_thread_data_t thread_data[PCIE_NUM_STREAMS];
    pthread_t threads[PCIE_NUM_STREAMS];

    int alloc_success = 1;
    for (int i = 0; i < num_streams; i++) {
        err = cudaHostAlloc(&thread_data[i].h_data, size, cudaHostAllocDefault);
        if (err != cudaSuccess) { alloc_success = 0; break; }

        err = cudaMalloc(&thread_data[i].d_data, size);
        if (err != cudaSuccess) {
            cudaFreeHost(thread_data[i].h_data);
            alloc_success = 0;
            break;
        }

        memset(thread_data[i].h_data, i + 1, size);
        cudaStreamCreate(&thread_data[i].stream);
        thread_data[i].size = size;
        thread_data[i].iterations = 5;
        thread_data[i].direction = i % 2;  /* Alternate H2D and D2H */
    }

    if (!alloc_success) {
        /* Cleanup any allocated resources */
        for (int i = 0; i < num_streams; i++) {
            if (thread_data[i].h_data) cudaFreeHost(thread_data[i].h_data);
            if (thread_data[i].d_data) cudaFree(thread_data[i].d_data);
        }
        result.success = 0;
        snprintf(result.error_message, sizeof(result.error_message),
                 "Failed to allocate memory for contention test");
        return result;
    }

    double contention_impacts[PCIE_TEST_ITERATIONS];
    double total_impact = 0.0;

    for (int iter = 0; iter < PCIE_TEST_ITERATIONS; iter++) {
        /* Launch all transfers concurrently */
        for (int i = 0; i < num_streams; i++) {
            pthread_create(&threads[i], NULL, pcie_worker_thread, &thread_data[i]);
        }

        /* Wait for all */
        for (int i = 0; i < num_streams; i++) {
            pthread_join(threads[i], NULL);
        }

        /* Calculate contention impact for tenant 0 */
        double contention_impact = ((baseline_bw - thread_data[0].bandwidth) / baseline_bw) * 100.0;
        if (contention_impact < 0) contention_impact = 0;

        contention_impacts[iter] = contention_impact;
        total_impact += contention_impact;
    }

    /* Cleanup */
    for (int i = 0; i < num_streams; i++) {
        cudaFree(thread_data[i].d_data);
        cudaFreeHost(thread_data[i].h_data);
        cudaStreamDestroy(thread_data[i].stream);
    }

    result.value = total_impact / PCIE_TEST_ITERATIONS;

    /* Calculate stddev */
    double variance = 0.0;
    for (int i = 0; i < PCIE_TEST_ITERATIONS; i++) {
        double diff = contention_impacts[i] - result.value;
        variance += diff * diff;
    }
    result.stddev = sqrt(variance / PCIE_TEST_ITERATIONS);

    result.success = 1;
    result.iterations = PCIE_TEST_ITERATIONS;
    snprintf(result.details, sizeof(result.details),
             "Baseline=%.2f GB/s, %d concurrent streams, impact=%.1f%%",
             baseline_bw, num_streams, result.value);

    return result;
}

/*
 * PCIE-004: Pinned Memory Performance Ratio
 * Measures performance ratio of pinned vs pageable memory transfers
 */
static bench_result_t bench_pcie_pinned_memory(bench_config_t *config) {
    bench_result_t result;
    memset(&result, 0, sizeof(result));
    strncpy(result.metric_id, METRIC_PCIE_PINNED_MEMORY, sizeof(result.metric_id) - 1);
    strncpy(result.name, "Pinned/Pageable Ratio", sizeof(result.name) - 1);
    strncpy(result.unit, "x", sizeof(result.unit) - 1);

    size_t size = 64 * 1024 * 1024;  /* 64 MB */

    /* Allocate pinned memory */
    void *h_pinned;
    cudaError_t err = cudaHostAlloc(&h_pinned, size, cudaHostAllocDefault);
    if (err != cudaSuccess) {
        result.success = 0;
        snprintf(result.error_message, sizeof(result.error_message),
                 "Failed to allocate pinned memory");
        return result;
    }
    memset(h_pinned, 0xAB, size);

    /* Allocate pageable memory */
    void *h_pageable = malloc(size);
    if (h_pageable == NULL) {
        cudaFreeHost(h_pinned);
        result.success = 0;
        snprintf(result.error_message, sizeof(result.error_message),
                 "Failed to allocate pageable memory");
        return result;
    }
    memset(h_pageable, 0xAB, size);

    void *d_data;
    err = cudaMalloc(&d_data, size);
    if (err != cudaSuccess) {
        cudaFreeHost(h_pinned);
        free(h_pageable);
        result.success = 0;
        snprintf(result.error_message, sizeof(result.error_message),
                 "Failed to allocate device memory");
        return result;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    double ratios[PCIE_TEST_ITERATIONS];
    double total_ratio = 0.0;

    for (int i = 0; i < PCIE_TEST_ITERATIONS; i++) {
        /* Measure pinned memory bandwidth */
        double pinned_bw = measure_h2d_bandwidth(h_pinned, d_data, size, stream, 5);

        /* Measure pageable memory bandwidth (uses cudaMemcpy which is sync) */
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);
        for (int j = 0; j < 5; j++) {
            cudaMemcpy(d_data, h_pageable, size, cudaMemcpyHostToDevice);
        }
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, start, stop);

        double pageable_bw = ((double)size * 5.0) / (elapsed_ms / 1000.0) / 1e9;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        /* Ratio of pinned to pageable performance */
        double ratio = (pageable_bw > 0) ? pinned_bw / pageable_bw : 1.0;
        ratios[i] = ratio;
        total_ratio += ratio;
    }

    cudaStreamDestroy(stream);
    cudaFree(d_data);
    cudaFreeHost(h_pinned);
    free(h_pageable);

    result.value = total_ratio / PCIE_TEST_ITERATIONS;

    /* Calculate stddev */
    double variance = 0.0;
    for (int i = 0; i < PCIE_TEST_ITERATIONS; i++) {
        double diff = ratios[i] - result.value;
        variance += diff * diff;
    }
    result.stddev = sqrt(variance / PCIE_TEST_ITERATIONS);

    result.success = 1;
    result.iterations = PCIE_TEST_ITERATIONS;
    snprintf(result.details, sizeof(result.details),
             "Pinned vs pageable memory transfer performance");

    return result;
}

/*
 * Run all PCIe benchmarks
 */
extern "C" void bench_run_pcie(bench_config_t *config, bench_result_t *results, int *count) {
    int idx = 0;

    printf("\n=== PCIe Bandwidth Metrics ===\n\n");

    /* PCIE-001: Host-to-Device Bandwidth */
    printf("Running PCIE-001: Host-to-Device Bandwidth...\n");
    results[idx] = bench_pcie_h2d_bandwidth(config);
    if (results[idx].success) {
        printf("  Result: %.2f %s (+/- %.2f)\n",
               results[idx].value, results[idx].unit, results[idx].stddev);
        printf("  %s\n", results[idx].details);
    } else {
        printf("  FAILED: %s\n", results[idx].error_message);
    }
    idx++;

    /* PCIE-002: Device-to-Host Bandwidth */
    printf("Running PCIE-002: Device-to-Host Bandwidth...\n");
    results[idx] = bench_pcie_d2h_bandwidth(config);
    if (results[idx].success) {
        printf("  Result: %.2f %s (+/- %.2f)\n",
               results[idx].value, results[idx].unit, results[idx].stddev);
        printf("  %s\n", results[idx].details);
    } else {
        printf("  FAILED: %s\n", results[idx].error_message);
    }
    idx++;

    /* PCIE-003: PCIe Contention Impact */
    printf("Running PCIE-003: PCIe Contention Impact...\n");
    results[idx] = bench_pcie_contention(config);
    if (results[idx].success) {
        printf("  Result: %.2f %s (+/- %.2f)\n",
               results[idx].value, results[idx].unit, results[idx].stddev);
        printf("  %s\n", results[idx].details);
    } else {
        printf("  FAILED: %s\n", results[idx].error_message);
    }
    idx++;

    /* PCIE-004: Pinned Memory Performance */
    printf("Running PCIE-004: Pinned Memory Performance...\n");
    results[idx] = bench_pcie_pinned_memory(config);
    if (results[idx].success) {
        printf("  Result: %.2f %s (+/- %.2f)\n",
               results[idx].value, results[idx].unit, results[idx].stddev);
        printf("  %s\n", results[idx].details);
    } else {
        printf("  FAILED: %s\n", results[idx].error_message);
    }
    idx++;

    *count = idx;
    printf("\nPCIe metrics completed: %d tests\n", idx);
}
