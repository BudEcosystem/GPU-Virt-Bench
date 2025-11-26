/*
 * Memory Bandwidth Isolation Metrics (BW-001 to BW-004)
 *
 * Measures memory bandwidth isolation and fairness under GPU virtualization.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <cuda_runtime.h>

extern "C" {
#include "include/benchmark.h"
#include "include/metrics.h"
}

/* Configuration */
#define BW_WARMUP_ITERATIONS    5
#define BW_TEST_ITERATIONS      20
#define BW_TEST_SIZE           (256 * 1024 * 1024)  /* 256 MB */

/* Bandwidth copy kernel */
__global__ void bandwidth_copy_kernel(float *dst, const float *src, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < n; i += stride) {
        dst[i] = src[i];
    }
}

/* Get theoretical memory bandwidth in GB/s */
static double get_theoretical_bandwidth(void) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    /* Memory bandwidth = clock * bus width * 2 (for DDR) / 8 (bits to bytes) / 1e9 */
    double bw = (double)prop.memoryClockRate * 1000.0 *
                (double)prop.memoryBusWidth / 8.0 * 2.0 / 1e9;
    return bw;
}

/* Measure memory bandwidth in GB/s */
static double measure_bandwidth(float *d_src, float *d_dst, size_t size,
                                cudaStream_t stream, int iterations) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    size_t num_elements = size / sizeof(float);
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    /* Warmup */
    bandwidth_copy_kernel<<<blocks, threads, 0, stream>>>(d_dst, d_src, num_elements);
    cudaStreamSynchronize(stream);

    /* Measure */
    cudaEventRecord(start, stream);
    for (int i = 0; i < iterations; i++) {
        bandwidth_copy_kernel<<<blocks, threads, 0, stream>>>(d_dst, d_src, num_elements);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    /* Calculate bandwidth: bytes transferred / time */
    /* Both read and write = 2 * size per iteration */
    double bytes_transferred = 2.0 * size * iterations;
    double time_seconds = elapsed_ms / 1000.0;
    double bandwidth_gbps = (bytes_transferred / time_seconds) / (1024.0 * 1024.0 * 1024.0);

    return bandwidth_gbps;
}

/*
 * BW-001: Memory Bandwidth Isolation
 *
 * Measures achieved bandwidth as percentage of theoretical maximum.
 */
static bench_result_t bench_bandwidth_isolation(bench_config_t *config) {
    bench_result_t result;
    memset(&result, 0, sizeof(result));
    strncpy(result.metric_id, METRIC_BW_ISOLATION, sizeof(result.metric_id) - 1);
    strncpy(result.name, "Memory Bandwidth Isolation", sizeof(result.name) - 1);
    strncpy(result.unit, "%", sizeof(result.unit) - 1);

    size_t size = BW_TEST_SIZE;
    float *d_src, *d_dst;

    cudaError_t err = cudaMalloc(&d_src, size);
    if (err != cudaSuccess) {
        result.success = 0;
        snprintf(result.error_message, sizeof(result.error_message),
                 "Failed to allocate source memory");
        return result;
    }

    err = cudaMalloc(&d_dst, size);
    if (err != cudaSuccess) {
        cudaFree(d_src);
        result.success = 0;
        snprintf(result.error_message, sizeof(result.error_message),
                 "Failed to allocate destination memory");
        return result;
    }

    cudaMemset(d_src, 1, size);
    cudaMemset(d_dst, 0, size);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    double theoretical_bw = get_theoretical_bandwidth();
    double measurements[BW_TEST_ITERATIONS];
    double total = 0.0;

    for (int iter = 0; iter < BW_TEST_ITERATIONS; iter++) {
        double bw = measure_bandwidth(d_src, d_dst, size, stream, 5);
        double isolation_pct = (bw / theoretical_bw) * 100.0;
        if (isolation_pct > 100.0) isolation_pct = 100.0;
        measurements[iter] = isolation_pct;
        total += isolation_pct;
    }

    result.value = total / BW_TEST_ITERATIONS;

    /* Calculate stddev */
    double variance = 0.0;
    for (int i = 0; i < BW_TEST_ITERATIONS; i++) {
        double diff = measurements[i] - result.value;
        variance += diff * diff;
    }
    result.stddev = sqrt(variance / BW_TEST_ITERATIONS);

    result.success = 1;
    result.iterations = BW_TEST_ITERATIONS;
    snprintf(result.details, sizeof(result.details),
             "Achieved %.1f%% of theoretical %.1f GB/s",
             result.value, theoretical_bw);

    cudaStreamDestroy(stream);
    cudaFree(d_src);
    cudaFree(d_dst);

    return result;
}

/*
 * BW-002: Bandwidth Fairness Index
 *
 * Measures bandwidth fairness across multiple concurrent streams.
 */
static bench_result_t bench_bandwidth_fairness(bench_config_t *config) {
    bench_result_t result;
    memset(&result, 0, sizeof(result));
    strncpy(result.metric_id, METRIC_BW_FAIRNESS, sizeof(result.metric_id) - 1);
    strncpy(result.name, "Bandwidth Fairness Index", sizeof(result.name) - 1);
    strncpy(result.unit, "ratio", sizeof(result.unit) - 1);

    int num_streams = 4;
    size_t size_per_stream = 64 * 1024 * 1024;  /* 64 MB per stream */

    float **d_src = (float**)malloc(num_streams * sizeof(float*));
    float **d_dst = (float**)malloc(num_streams * sizeof(float*));
    cudaStream_t *streams = (cudaStream_t*)malloc(num_streams * sizeof(cudaStream_t));
    cudaEvent_t *starts = (cudaEvent_t*)malloc(num_streams * sizeof(cudaEvent_t));
    cudaEvent_t *stops = (cudaEvent_t*)malloc(num_streams * sizeof(cudaEvent_t));

    /* Allocate resources for each stream */
    for (int s = 0; s < num_streams; s++) {
        cudaError_t err = cudaMalloc(&d_src[s], size_per_stream);
        if (err != cudaSuccess) {
            for (int j = 0; j < s; j++) {
                cudaFree(d_src[j]);
                cudaFree(d_dst[j]);
                cudaStreamDestroy(streams[j]);
            }
            free(d_src); free(d_dst); free(streams); free(starts); free(stops);
            result.success = 0;
            snprintf(result.error_message, sizeof(result.error_message),
                     "Failed to allocate memory for stream %d", s);
            return result;
        }
        cudaMalloc(&d_dst[s], size_per_stream);
        cudaStreamCreate(&streams[s]);
        cudaEventCreate(&starts[s]);
        cudaEventCreate(&stops[s]);
        cudaMemset(d_src[s], s, size_per_stream);
    }

    size_t num_elements = size_per_stream / sizeof(float);
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    double fairness_values[BW_TEST_ITERATIONS];
    double total_fairness = 0.0;

    for (int iter = 0; iter < BW_TEST_ITERATIONS; iter++) {
        double bandwidths[4];

        /* Launch all streams concurrently */
        for (int s = 0; s < num_streams; s++) {
            cudaEventRecord(starts[s], streams[s]);
            for (int rep = 0; rep < 10; rep++) {
                bandwidth_copy_kernel<<<blocks, threads, 0, streams[s]>>>(
                    d_dst[s], d_src[s], num_elements);
            }
            cudaEventRecord(stops[s], streams[s]);
        }

        /* Wait and collect timings */
        for (int s = 0; s < num_streams; s++) {
            cudaStreamSynchronize(streams[s]);
            float elapsed_ms;
            cudaEventElapsedTime(&elapsed_ms, starts[s], stops[s]);
            double bytes = 2.0 * size_per_stream * 10;
            bandwidths[s] = (bytes / (elapsed_ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0);
        }

        /* Calculate Jain's fairness index */
        double sum = 0.0, sum_sq = 0.0;
        for (int s = 0; s < num_streams; s++) {
            sum += bandwidths[s];
            sum_sq += bandwidths[s] * bandwidths[s];
        }
        double fairness = (sum * sum) / (num_streams * sum_sq);
        fairness_values[iter] = fairness;
        total_fairness += fairness;
    }

    result.value = total_fairness / BW_TEST_ITERATIONS;

    /* Calculate stddev */
    double variance = 0.0;
    for (int i = 0; i < BW_TEST_ITERATIONS; i++) {
        double diff = fairness_values[i] - result.value;
        variance += diff * diff;
    }
    result.stddev = sqrt(variance / BW_TEST_ITERATIONS);

    result.success = 1;
    result.iterations = BW_TEST_ITERATIONS;
    snprintf(result.details, sizeof(result.details),
             "Jain's fairness index across %d streams: %.3f",
             num_streams, result.value);

    /* Cleanup */
    for (int s = 0; s < num_streams; s++) {
        cudaFree(d_src[s]);
        cudaFree(d_dst[s]);
        cudaStreamDestroy(streams[s]);
        cudaEventDestroy(starts[s]);
        cudaEventDestroy(stops[s]);
    }
    free(d_src); free(d_dst); free(streams); free(starts); free(stops);

    return result;
}

/*
 * BW-003: Memory Bus Saturation
 *
 * Measures how many concurrent memory streams saturate the memory bus.
 */
static bench_result_t bench_bandwidth_saturation(bench_config_t *config) {
    bench_result_t result;
    memset(&result, 0, sizeof(result));
    strncpy(result.metric_id, METRIC_BW_SATURATION, sizeof(result.metric_id) - 1);
    strncpy(result.name, "Memory Bus Saturation Point", sizeof(result.name) - 1);
    strncpy(result.unit, "streams", sizeof(result.unit) - 1);

    size_t size = 64 * 1024 * 1024;  /* 64 MB per stream */
    int max_streams = 8;
    double baseline_bw = 0.0;
    int saturation_point = 1;

    float *d_src, *d_dst;
    cudaError_t err = cudaMalloc(&d_src, size);
    if (err != cudaSuccess) {
        result.success = 0;
        snprintf(result.error_message, sizeof(result.error_message),
                 "Failed to allocate memory");
        return result;
    }
    cudaMalloc(&d_dst, size);
    cudaMemset(d_src, 1, size);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    /* Measure baseline with single stream */
    baseline_bw = measure_bandwidth(d_src, d_dst, size, stream, 10);

    cudaStreamDestroy(stream);
    cudaFree(d_src);
    cudaFree(d_dst);

    /* Test with increasing number of streams */
    for (int num_streams = 2; num_streams <= max_streams; num_streams++) {
        float **d_srcs = (float**)malloc(num_streams * sizeof(float*));
        float **d_dsts = (float**)malloc(num_streams * sizeof(float*));
        cudaStream_t *streams = (cudaStream_t*)malloc(num_streams * sizeof(cudaStream_t));

        bool alloc_ok = true;
        for (int s = 0; s < num_streams && alloc_ok; s++) {
            if (cudaMalloc(&d_srcs[s], size) != cudaSuccess) alloc_ok = false;
            if (cudaMalloc(&d_dsts[s], size) != cudaSuccess) alloc_ok = false;
            cudaStreamCreate(&streams[s]);
            if (alloc_ok) cudaMemset(d_srcs[s], s, size);
        }

        if (!alloc_ok) {
            for (int j = 0; j < num_streams; j++) {
                if (d_srcs[j]) cudaFree(d_srcs[j]);
                if (d_dsts[j]) cudaFree(d_dsts[j]);
                cudaStreamDestroy(streams[j]);
            }
            free(d_srcs); free(d_dsts); free(streams);
            break;
        }

        size_t num_elements = size / sizeof(float);
        int threads = 256;
        int blocks = (num_elements + threads - 1) / threads;
        if (blocks > 65535) blocks = 65535;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        /* Measure total bandwidth with all streams active */
        cudaEventRecord(start, streams[0]);
        for (int rep = 0; rep < 10; rep++) {
            for (int s = 0; s < num_streams; s++) {
                bandwidth_copy_kernel<<<blocks, threads, 0, streams[s]>>>(
                    d_dsts[s], d_srcs[s], num_elements);
            }
        }
        for (int s = 0; s < num_streams; s++) {
            cudaStreamSynchronize(streams[s]);
        }
        cudaEventRecord(stop, streams[0]);
        cudaEventSynchronize(stop);

        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, start, stop);

        double total_bytes = 2.0 * size * num_streams * 10;
        double total_bw = (total_bytes / (elapsed_ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0);

        /* Check if adding streams increases total bandwidth */
        if (total_bw > baseline_bw * 1.05) {  /* 5% improvement threshold */
            saturation_point = num_streams;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        for (int s = 0; s < num_streams; s++) {
            cudaFree(d_srcs[s]);
            cudaFree(d_dsts[s]);
            cudaStreamDestroy(streams[s]);
        }
        free(d_srcs); free(d_dsts); free(streams);
    }

    result.value = saturation_point;
    result.stddev = 0.0;
    result.success = 1;
    result.iterations = max_streams;
    snprintf(result.details, sizeof(result.details),
             "Memory bus saturates at %d concurrent streams",
             saturation_point);

    return result;
}

/*
 * BW-004: Bandwidth Interference
 *
 * Measures bandwidth degradation under concurrent memory access.
 */
static bench_result_t bench_bandwidth_interference(bench_config_t *config) {
    bench_result_t result;
    memset(&result, 0, sizeof(result));
    strncpy(result.metric_id, METRIC_BW_INTERFERENCE, sizeof(result.metric_id) - 1);
    strncpy(result.name, "Bandwidth Interference Impact", sizeof(result.name) - 1);
    strncpy(result.unit, "%", sizeof(result.unit) - 1);

    size_t size = 128 * 1024 * 1024;  /* 128 MB */
    int num_interferers = 3;

    float *d_src, *d_dst;
    float **d_int_src = (float**)malloc(num_interferers * sizeof(float*));
    float **d_int_dst = (float**)malloc(num_interferers * sizeof(float*));
    cudaStream_t main_stream;
    cudaStream_t *int_streams = (cudaStream_t*)malloc(num_interferers * sizeof(cudaStream_t));

    cudaError_t err = cudaMalloc(&d_src, size);
    if (err != cudaSuccess) {
        free(d_int_src); free(d_int_dst); free(int_streams);
        result.success = 0;
        snprintf(result.error_message, sizeof(result.error_message),
                 "Failed to allocate main memory");
        return result;
    }
    cudaMalloc(&d_dst, size);
    cudaStreamCreate(&main_stream);
    cudaMemset(d_src, 1, size);

    for (int i = 0; i < num_interferers; i++) {
        cudaMalloc(&d_int_src[i], size);
        cudaMalloc(&d_int_dst[i], size);
        cudaStreamCreate(&int_streams[i]);
        cudaMemset(d_int_src[i], i, size);
    }

    /* Measure baseline bandwidth (no interference) */
    double baseline_bw = measure_bandwidth(d_src, d_dst, size, main_stream, 10);

    /* Measure bandwidth under interference */
    size_t num_elements = size / sizeof(float);
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    double interference_measurements[BW_TEST_ITERATIONS];
    double total_interference = 0.0;

    for (int iter = 0; iter < BW_TEST_ITERATIONS; iter++) {
        /* Start interferers */
        for (int i = 0; i < num_interferers; i++) {
            for (int rep = 0; rep < 20; rep++) {
                bandwidth_copy_kernel<<<blocks, threads, 0, int_streams[i]>>>(
                    d_int_dst[i], d_int_src[i], num_elements);
            }
        }

        /* Measure main stream bandwidth */
        double interfered_bw = measure_bandwidth(d_src, d_dst, size, main_stream, 5);

        /* Wait for interferers */
        for (int i = 0; i < num_interferers; i++) {
            cudaStreamSynchronize(int_streams[i]);
        }

        double degradation = ((baseline_bw - interfered_bw) / baseline_bw) * 100.0;
        if (degradation < 0) degradation = 0;
        interference_measurements[iter] = degradation;
        total_interference += degradation;
    }

    result.value = total_interference / BW_TEST_ITERATIONS;

    /* Calculate stddev */
    double variance = 0.0;
    for (int i = 0; i < BW_TEST_ITERATIONS; i++) {
        double diff = interference_measurements[i] - result.value;
        variance += diff * diff;
    }
    result.stddev = sqrt(variance / BW_TEST_ITERATIONS);

    result.success = 1;
    result.iterations = BW_TEST_ITERATIONS;
    snprintf(result.details, sizeof(result.details),
             "%.1f%% bandwidth degradation with %d interferers",
             result.value, num_interferers);

    /* Cleanup */
    cudaStreamDestroy(main_stream);
    cudaFree(d_src);
    cudaFree(d_dst);
    for (int i = 0; i < num_interferers; i++) {
        cudaFree(d_int_src[i]);
        cudaFree(d_int_dst[i]);
        cudaStreamDestroy(int_streams[i]);
    }
    free(d_int_src); free(d_int_dst); free(int_streams);

    return result;
}

/*
 * Run all bandwidth metrics
 */
extern "C" void bench_run_bandwidth(bench_config_t *config, bench_result_t *results, int *count) {
    int idx = 0;

    printf("\n=== Memory Bandwidth Isolation Metrics ===\n\n");

    /* BW-001: Bandwidth Isolation */
    printf("Running BW-001: Memory Bandwidth Isolation...\n");
    results[idx] = bench_bandwidth_isolation(config);
    if (results[idx].success) {
        printf("  Result: %.2f %s (+/- %.2f)\n",
               results[idx].value, results[idx].unit, results[idx].stddev);
        printf("  %s\n", results[idx].details);
    } else {
        printf("  FAILED: %s\n", results[idx].error_message);
    }
    idx++;

    /* BW-002: Bandwidth Fairness */
    printf("Running BW-002: Bandwidth Fairness Index...\n");
    results[idx] = bench_bandwidth_fairness(config);
    if (results[idx].success) {
        printf("  Result: %.3f %s (+/- %.3f)\n",
               results[idx].value, results[idx].unit, results[idx].stddev);
        printf("  %s\n", results[idx].details);
    } else {
        printf("  FAILED: %s\n", results[idx].error_message);
    }
    idx++;

    /* BW-003: Bandwidth Saturation */
    printf("Running BW-003: Memory Bus Saturation Point...\n");
    results[idx] = bench_bandwidth_saturation(config);
    if (results[idx].success) {
        printf("  Result: %.0f %s\n",
               results[idx].value, results[idx].unit);
        printf("  %s\n", results[idx].details);
    } else {
        printf("  FAILED: %s\n", results[idx].error_message);
    }
    idx++;

    /* BW-004: Bandwidth Interference */
    printf("Running BW-004: Bandwidth Interference Impact...\n");
    results[idx] = bench_bandwidth_interference(config);
    if (results[idx].success) {
        printf("  Result: %.2f %s (+/- %.2f)\n",
               results[idx].value, results[idx].unit, results[idx].stddev);
        printf("  %s\n", results[idx].details);
    } else {
        printf("  FAILED: %s\n", results[idx].error_message);
    }
    idx++;

    *count = idx;
    printf("\nBandwidth metrics completed: %d tests\n", idx);
}
