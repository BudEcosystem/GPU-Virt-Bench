/*
 * Error Recovery Metrics (ERR-001 to ERR-003)
 *
 * Measures GPU error handling and recovery capabilities.
 * Critical for production systems where reliability matters.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

extern "C" {
#include "include/benchmark.h"
#include "include/metrics.h"
}

/* Configuration */
#define ERR_WARMUP_ITERATIONS    3
#define ERR_TEST_ITERATIONS      20
#define ERR_RECOVERY_ATTEMPTS    10

/* Get current time in microseconds */
static double get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1000.0;
}

/* Simple kernel that always succeeds */
__global__ void success_kernel(float *data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] + 1.0f;
    }
}

/* Kernel for stress testing */
__global__ void stress_kernel(float *data, size_t n, int iterations) {
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

/*
 * ERR-001: Error Detection Latency
 *
 * Measures how quickly CUDA errors are detected and reported.
 */
static bench_result_t bench_error_detection_latency(bench_config_t *config) {
    bench_result_t result;
    memset(&result, 0, sizeof(result));
    strncpy(result.metric_id, METRIC_ERR_DETECTION, sizeof(result.metric_id) - 1);
    strncpy(result.name, "Error Detection Latency", sizeof(result.name) - 1);
    strncpy(result.unit, "us", sizeof(result.unit) - 1);

    double detection_times[ERR_TEST_ITERATIONS];
    double total_time = 0.0;
    int valid_measurements = 0;

    for (int iter = 0; iter < ERR_TEST_ITERATIONS; iter++) {
        /* Clear any previous errors */
        cudaGetLastError();

        /* Try to allocate an impossibly large amount of memory */
        void *ptr = NULL;
        size_t huge_size = 1024ULL * 1024 * 1024 * 1024;  /* 1 TB */

        double t1 = get_time_us();
        cudaError_t err = cudaMalloc(&ptr, huge_size);
        double t2 = get_time_us();

        if (err != cudaSuccess) {
            /* Error was detected */
            detection_times[valid_measurements] = t2 - t1;
            total_time += detection_times[valid_measurements];
            valid_measurements++;
        }

        /* Clear the error */
        cudaGetLastError();
    }

    if (valid_measurements == 0) {
        result.success = 0;
        snprintf(result.error_message, sizeof(result.error_message),
                 "Could not trigger allocation failures");
        return result;
    }

    result.value = total_time / valid_measurements;

    /* Calculate stddev */
    double variance = 0.0;
    for (int i = 0; i < valid_measurements; i++) {
        double diff = detection_times[i] - result.value;
        variance += diff * diff;
    }
    result.stddev = sqrt(variance / valid_measurements);

    result.success = 1;
    result.iterations = valid_measurements;
    snprintf(result.details, sizeof(result.details),
             "Allocation error detection, measurements=%d, latency=%.2f us",
             valid_measurements, result.value);

    return result;
}

/*
 * ERR-002: Recovery Time
 *
 * Measures time to recover GPU to a usable state after an error.
 */
static bench_result_t bench_recovery_time(bench_config_t *config) {
    bench_result_t result;
    memset(&result, 0, sizeof(result));
    strncpy(result.metric_id, METRIC_ERR_RECOVERY_TIME, sizeof(result.metric_id) - 1);
    strncpy(result.name, "Error Recovery Time", sizeof(result.name) - 1);
    strncpy(result.unit, "us", sizeof(result.unit) - 1);

    size_t test_size = 16 * 1024 * 1024;  /* 16 MB */
    size_t num_elements = test_size / sizeof(float);

    double recovery_times[ERR_TEST_ITERATIONS];
    double total_time = 0.0;
    int valid_measurements = 0;

    for (int iter = 0; iter < ERR_TEST_ITERATIONS; iter++) {
        /* First, induce an error condition */
        void *bad_ptr = NULL;
        size_t huge_size = 1024ULL * 1024 * 1024 * 1024;  /* 1 TB */
        cudaMalloc(&bad_ptr, huge_size);  /* This will fail */

        /* Get the error */
        cudaError_t err = cudaGetLastError();
        if (err == cudaSuccess) {
            continue;  /* No error to recover from */
        }

        /* Measure time to recover and perform a successful operation */
        double t1 = get_time_us();

        /* Clear the error state */
        cudaGetLastError();

        /* Attempt to perform normal GPU operations */
        float *d_data = NULL;
        err = cudaMalloc(&d_data, test_size);

        if (err == cudaSuccess) {
            cudaMemset(d_data, 0, test_size);

            int threads = 256;
            int blocks = (num_elements + threads - 1) / threads;
            if (blocks > 65535) blocks = 65535;

            success_kernel<<<blocks, threads>>>(d_data, num_elements);
            cudaDeviceSynchronize();

            err = cudaGetLastError();
            if (err == cudaSuccess) {
                double t2 = get_time_us();
                recovery_times[valid_measurements] = t2 - t1;
                total_time += recovery_times[valid_measurements];
                valid_measurements++;
            }

            cudaFree(d_data);
        }
    }

    if (valid_measurements == 0) {
        result.success = 0;
        snprintf(result.error_message, sizeof(result.error_message),
                 "Could not complete recovery measurements");
        return result;
    }

    result.value = total_time / valid_measurements;

    /* Calculate stddev */
    double variance = 0.0;
    for (int i = 0; i < valid_measurements; i++) {
        double diff = recovery_times[i] - result.value;
        variance += diff * diff;
    }
    result.stddev = sqrt(variance / valid_measurements);

    result.success = 1;
    result.iterations = valid_measurements;
    snprintf(result.details, sizeof(result.details),
             "Recovery after allocation error, measurements=%d, time=%.2f us",
             valid_measurements, result.value);

    return result;
}

/*
 * ERR-003: Graceful Degradation Score
 *
 * Measures how well the GPU handles resource exhaustion without crashing.
 * Higher score means better graceful degradation.
 */
static bench_result_t bench_graceful_degradation(bench_config_t *config) {
    bench_result_t result;
    memset(&result, 0, sizeof(result));
    strncpy(result.metric_id, METRIC_ERR_GRACEFUL_DEGRADATION, sizeof(result.metric_id) - 1);
    strncpy(result.name, "Graceful Degradation Score", sizeof(result.name) - 1);
    strncpy(result.unit, "%", sizeof(result.unit) - 1);

    size_t free_mem, total_mem;
    cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
    if (err != cudaSuccess) {
        result.success = 0;
        snprintf(result.error_message, sizeof(result.error_message),
                 "Failed to get memory info");
        return result;
    }

    /* Test factors for graceful degradation:
     * 1. Can we detect memory exhaustion without crashing?
     * 2. Can we recover and continue working?
     * 3. Do allocations fail gracefully?
     */

    int total_tests = 0;
    int passed_tests = 0;

    /* Test 1: Memory exhaustion detection */
    {
        total_tests++;
        void **ptrs = (void**)malloc(256 * sizeof(void*));
        int num_allocs = 0;
        size_t alloc_size = 64 * 1024 * 1024;  /* 64 MB chunks */

        /* Allocate until we run out of memory */
        while (num_allocs < 256) {
            err = cudaMalloc(&ptrs[num_allocs], alloc_size);
            if (err != cudaSuccess) {
                break;  /* Graceful failure */
            }
            num_allocs++;
        }

        /* Check that we got an error (not a crash) */
        if (err == cudaErrorMemoryAllocation || err == cudaErrorMemoryValueTooLarge) {
            passed_tests++;  /* Graceful failure detected */
        }

        /* Clean up */
        for (int i = 0; i < num_allocs; i++) {
            cudaFree(ptrs[i]);
        }
        free(ptrs);
        cudaGetLastError();  /* Clear error state */
    }

    /* Test 2: Recovery after exhaustion */
    {
        total_tests++;

        /* Exhaust memory */
        void **ptrs = (void**)malloc(256 * sizeof(void*));
        int num_allocs = 0;
        size_t alloc_size = 64 * 1024 * 1024;

        while (num_allocs < 256) {
            err = cudaMalloc(&ptrs[num_allocs], alloc_size);
            if (err != cudaSuccess) break;
            num_allocs++;
        }

        /* Free all memory */
        for (int i = 0; i < num_allocs; i++) {
            cudaFree(ptrs[i]);
        }
        free(ptrs);

        /* Clear error state */
        cudaGetLastError();

        /* Try to allocate and use memory again */
        float *d_test = NULL;
        err = cudaMalloc(&d_test, 16 * 1024 * 1024);
        if (err == cudaSuccess) {
            cudaMemset(d_test, 0, 16 * 1024 * 1024);
            cudaDeviceSynchronize();

            err = cudaGetLastError();
            if (err == cudaSuccess) {
                passed_tests++;  /* Successful recovery */
            }
            cudaFree(d_test);
        }
    }

    /* Test 3: Continued operation under pressure */
    {
        total_tests++;

        /* Allocate 80% of free memory */
        size_t current_free, current_total;
        cudaMemGetInfo(&current_free, &current_total);
        size_t pressure_size = (current_free * 80) / 100;

        void *pressure_ptr = NULL;
        err = cudaMalloc(&pressure_ptr, pressure_size);

        if (err == cudaSuccess) {
            /* Try to do work with remaining memory */
            size_t work_size = 4 * 1024 * 1024;  /* 4 MB */
            size_t num_elements = work_size / sizeof(float);
            float *d_work = NULL;

            err = cudaMalloc(&d_work, work_size);
            if (err == cudaSuccess) {
                cudaMemset(d_work, 0, work_size);

                int threads = 256;
                int blocks = (num_elements + threads - 1) / threads;
                if (blocks > 65535) blocks = 65535;

                success_kernel<<<blocks, threads>>>(d_work, num_elements);
                cudaDeviceSynchronize();

                err = cudaGetLastError();
                if (err == cudaSuccess) {
                    passed_tests++;  /* Continued operation successful */
                }
                cudaFree(d_work);
            }
            cudaFree(pressure_ptr);
        }
    }

    /* Test 4: Multiple recovery cycles */
    {
        total_tests++;
        int recovery_success = 0;

        for (int cycle = 0; cycle < 5; cycle++) {
            /* Induce error */
            void *bad = NULL;
            cudaMalloc(&bad, 1024ULL * 1024 * 1024 * 1024);
            cudaGetLastError();

            /* Try to recover */
            float *good = NULL;
            err = cudaMalloc(&good, 1024 * 1024);
            if (err == cudaSuccess) {
                cudaFree(good);
                recovery_success++;
            }
        }

        if (recovery_success == 5) {
            passed_tests++;  /* All recovery cycles succeeded */
        }
    }

    /* Test 5: Error isolation between operations */
    {
        total_tests++;

        /* Create stream for isolation test */
        cudaStream_t stream1, stream2;
        cudaStreamCreate(&stream1);
        cudaStreamCreate(&stream2);

        float *d_data1 = NULL, *d_data2 = NULL;
        size_t test_size = 8 * 1024 * 1024;
        size_t num_elements = test_size / sizeof(float);

        int success = 1;

        err = cudaMalloc(&d_data1, test_size);
        if (err != cudaSuccess) success = 0;

        err = cudaMalloc(&d_data2, test_size);
        if (err != cudaSuccess) success = 0;

        if (success) {
            cudaMemset(d_data1, 0, test_size);
            cudaMemset(d_data2, 0, test_size);

            int threads = 256;
            int blocks = (num_elements + threads - 1) / threads;
            if (blocks > 65535) blocks = 65535;

            /* Launch kernel on stream1 */
            success_kernel<<<blocks, threads, 0, stream1>>>(d_data1, num_elements);

            /* Induce an allocation error */
            void *bad = NULL;
            cudaMalloc(&bad, 1024ULL * 1024 * 1024 * 1024);
            cudaGetLastError();

            /* Launch kernel on stream2 - should still work */
            success_kernel<<<blocks, threads, 0, stream2>>>(d_data2, num_elements);

            cudaStreamSynchronize(stream1);
            cudaStreamSynchronize(stream2);

            err = cudaGetLastError();
            if (err == cudaSuccess) {
                passed_tests++;  /* Operations isolated correctly */
            }

            cudaFree(d_data1);
            cudaFree(d_data2);
        }

        cudaStreamDestroy(stream1);
        cudaStreamDestroy(stream2);
    }

    /* Calculate score */
    result.value = ((double)passed_tests / (double)total_tests) * 100.0;
    result.stddev = 0.0;  /* Single measurement */

    result.success = 1;
    result.iterations = total_tests;
    snprintf(result.details, sizeof(result.details),
             "Graceful degradation tests: %d/%d passed (%.1f%%)",
             passed_tests, total_tests, result.value);

    return result;
}

/*
 * Run all error recovery metrics
 */
extern "C" void bench_run_error(bench_config_t *config, bench_result_t *results, int *count) {
    int idx = 0;

    printf("\n=== Error Recovery Metrics ===\n\n");

    /* ERR-001: Error Detection Latency */
    printf("Running ERR-001: Error Detection Latency...\n");
    results[idx] = bench_error_detection_latency(config);
    if (results[idx].success) {
        printf("  Result: %.2f %s (+/- %.2f)\n",
               results[idx].value, results[idx].unit, results[idx].stddev);
        printf("  %s\n", results[idx].details);
    } else {
        printf("  FAILED: %s\n", results[idx].error_message);
    }
    idx++;

    /* ERR-002: Recovery Time */
    printf("Running ERR-002: Recovery Time...\n");
    results[idx] = bench_recovery_time(config);
    if (results[idx].success) {
        printf("  Result: %.2f %s (+/- %.2f)\n",
               results[idx].value, results[idx].unit, results[idx].stddev);
        printf("  %s\n", results[idx].details);
    } else {
        printf("  FAILED: %s\n", results[idx].error_message);
    }
    idx++;

    /* ERR-003: Graceful Degradation Score */
    printf("Running ERR-003: Graceful Degradation Score...\n");
    results[idx] = bench_graceful_degradation(config);
    if (results[idx].success) {
        printf("  Result: %.2f %s\n",
               results[idx].value, results[idx].unit);
        printf("  %s\n", results[idx].details);
    } else {
        printf("  FAILED: %s\n", results[idx].error_message);
    }
    idx++;

    *count = idx;
    printf("\nError recovery metrics completed: %d tests\n", idx);
}
