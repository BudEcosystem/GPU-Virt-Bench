/**
 * Idle Stream Detector Benchmark for gpu-virt-bench
 *
 * Tests the UVM idle stream detector module for:
 * - Detection accuracy
 * - Thread-safety
 * - Performance overhead
 * - Callback latency
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "../include/benchmark.h"

// Include idle detector from bud_fcsp (included via CMake include_directories)
#include "idle_detector.h"

/*=============================================================================
 * Helper Kernels
 *=============================================================================*/

__global__ void short_compute_kernel(float *data, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = data[idx];

    for (int i = 0; i < iterations; i++) {
        val = sqrtf(val * 1.01f + 0.01f);
    }

    data[idx] = val;
}

__global__ void medium_compute_kernel(float *data, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = data[idx];

    for (int i = 0; i < iterations; i++) {
        val = sqrtf(val * 1.01f + 0.01f);
        val = expf(-val * 0.1f);
    }

    data[idx] = val;
}

/*=============================================================================
 * IDLE-001: Idle Detection Accuracy
 * Measures: Accuracy of detecting idle vs active streams
 * Target: >99% accuracy
 *=============================================================================*/

metric_result_t benchmark_idle_detection_accuracy(benchmark_config_t *config) {
    metric_result_t result = {0};
    strcpy(result.metric_id, "IDLE-001");
    strcpy(result.name, "Idle Detection Accuracy");
    strcpy(result.unit, "%");

    const int num_streams = 10;
    const size_t data_size = 1024 * 1024; // 1MB per stream
    const uint64_t idle_threshold_ms = 50;

    // Initialize idle detector
    uvm_idle_detector_t *detector = uvm_idle_detector_init(0, idle_threshold_ms);
    if (!detector) {
        result.success = false;
        strncpy(result.error_message, "Failed to initialize idle detector", sizeof(result.error_message) - 1);
        return result;
    }

    // Create streams and allocations
    cudaStream_t *streams = (cudaStream_t*)malloc(num_streams * sizeof(cudaStream_t));
    float **data = (float**)malloc(num_streams * sizeof(float*));

    if (!streams || !data) {
        result.success = false;
        strncpy(result.error_message, "Failed to allocate host memory", sizeof(result.error_message) - 1);
        uvm_idle_detector_shutdown(detector);
        return result;
    }

    // Setup
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
        cudaMalloc(&data[i], data_size * sizeof(float));
    }

    // Test 1: Launch kernels and immediately check (should NOT be idle)
    int correct_active = 0;
    for (int i = 0; i < num_streams; i++) {
        short_compute_kernel<<<256, 256, 0, streams[i]>>>(data[i], 1000);
        uvm_idle_detector_record_launch(detector, streams[i]);

        bool is_idle = uvm_idle_detector_is_idle(detector, streams[i]);
        if (!is_idle) {
            correct_active++;
        }
    }
    cudaDeviceSynchronize();

    // Test 2: Wait for idle threshold and check (should be idle)
    usleep((idle_threshold_ms + 10) * 1000); // Wait 60ms

    int correct_idle = 0;
    for (int i = 0; i < num_streams; i++) {
        bool is_idle = uvm_idle_detector_is_idle(detector, streams[i]);
        if (is_idle) {
            correct_idle++;
        }
    }

    // Test 3: Reactivate and check (should NOT be idle)
    int correct_reactivated = 0;
    for (int i = 0; i < num_streams; i++) {
        short_compute_kernel<<<256, 256, 0, streams[i]>>>(data[i], 1000);
        uvm_idle_detector_record_launch(detector, streams[i]);

        bool is_idle = uvm_idle_detector_is_idle(detector, streams[i]);
        if (!is_idle) {
            correct_reactivated++;
        }
    }
    cudaDeviceSynchronize();

    // Calculate accuracy
    int total_checks = num_streams * 3; // 3 tests
    int correct_detections = correct_active + correct_idle + correct_reactivated;
    double accuracy = (double)correct_detections / total_checks * 100.0;

    // Cleanup
    for (int i = 0; i < num_streams; i++) {
        cudaFree(data[i]);
        cudaStreamDestroy(streams[i]);
    }
    free(streams);
    free(data);
    uvm_idle_detector_shutdown(detector);

    result.value = accuracy;
    result.success = accuracy >= 99.0;

    return result;
}

/*=============================================================================
 * IDLE-002: Detection Latency
 * Measures: Time between stream becoming idle and detection
 * Target: < 100ms
 *=============================================================================*/

metric_result_t benchmark_idle_detection_latency(benchmark_config_t *config) {
    metric_result_t result = {0};
    strcpy(result.metric_id, "IDLE-002");
    strcpy(result.name, "Idle Detection Latency");
    strcpy(result.unit, "ms");

    const uint64_t idle_threshold_ms = 50;
    const int num_iterations = 20;

    uvm_idle_detector_t *detector = uvm_idle_detector_init(0, idle_threshold_ms);
    if (!detector) {
        result.success = false;
        strncpy(result.error_message, "Failed to initialize idle detector", sizeof(result.error_message) - 1);
        return result;
    }

    cudaStream_t stream;
    float *data;
    cudaStreamCreate(&stream);
    cudaMalloc(&data, 1024 * 1024 * sizeof(float));

    double total_latency_ms = 0;
    int successful_detections = 0;

    struct timespec start, now;

    for (int iter = 0; iter < num_iterations; iter++) {
        // Launch kernel
        short_compute_kernel<<<256, 256, 0, stream>>>(data, 100);
        uvm_idle_detector_record_launch(detector, stream);
        cudaStreamSynchronize(stream);

        // Record when stream became idle (kernel finished)
        clock_gettime(CLOCK_MONOTONIC, &start);

        // Poll until detector marks it as idle
        bool detected = false;
        uint64_t check_start_ns = start.tv_sec * 1000000000ULL + start.tv_nsec;

        for (int attempt = 0; attempt < 200; attempt++) { // Max 200ms wait
            usleep(1000); // 1ms sleep

            if (uvm_idle_detector_is_idle(detector, stream)) {
                clock_gettime(CLOCK_MONOTONIC, &now);
                uint64_t now_ns = now.tv_sec * 1000000000ULL + now.tv_nsec;
                uint64_t elapsed_ns = now_ns - check_start_ns;

                double latency_ms = elapsed_ns / 1000000.0;
                total_latency_ms += latency_ms;
                successful_detections++;
                detected = true;
                break;
            }
        }

        if (!detected) {
            // Failed to detect within timeout
            continue;
        }
    }

    cudaFree(data);
    cudaStreamDestroy(stream);
    uvm_idle_detector_shutdown(detector);

    if (successful_detections == 0) {
        result.success = false;
        strncpy(result.error_message, "No successful detections", sizeof(result.error_message) - 1);
        return result;
    }

    double avg_latency_ms = total_latency_ms / successful_detections;

    result.value = avg_latency_ms;
    result.success = avg_latency_ms < 100.0;

    return result;
}

/*=============================================================================
 * IDLE-003: Multi-Stream Scalability
 * Measures: Detection accuracy with many concurrent streams
 * Target: >95% accuracy with 100 streams
 *=============================================================================*/

metric_result_t benchmark_idle_multistream_scalability(benchmark_config_t *config) {
    metric_result_t result = {0};
    strcpy(result.metric_id, "IDLE-003");
    strcpy(result.name, "Multi-Stream Scalability");
    strcpy(result.unit, "%");

    const int num_streams = 100;
    const size_t data_size = 512 * 1024; // 512KB per stream
    const uint64_t idle_threshold_ms = 50;

    uvm_idle_detector_t *detector = uvm_idle_detector_init(0, idle_threshold_ms);
    if (!detector) {
        result.success = false;
        strncpy(result.error_message, "Failed to initialize idle detector", sizeof(result.error_message) - 1);
        return result;
    }

    // Create many streams
    cudaStream_t *streams = (cudaStream_t*)malloc(num_streams * sizeof(cudaStream_t));
    float **data = (float**)malloc(num_streams * sizeof(float*));

    if (!streams || !data) {
        result.success = false;
        strncpy(result.error_message, "Failed to allocate host memory", sizeof(result.error_message) - 1);
        uvm_idle_detector_shutdown(detector);
        return result;
    }

    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
        cudaMalloc(&data[i], data_size * sizeof(float));
    }

    // Launch on all streams
    for (int i = 0; i < num_streams; i++) {
        short_compute_kernel<<<128, 128, 0, streams[i]>>>(data[i], 500);
        uvm_idle_detector_record_launch(detector, streams[i]);
    }
    cudaDeviceSynchronize();

    // Wait for idle
    usleep((idle_threshold_ms + 10) * 1000);

    // Count idle streams using detector API
    cudaStream_t *idle_streams = (cudaStream_t*)malloc(num_streams * sizeof(cudaStream_t));
    int idle_count = uvm_idle_detector_get_idle_streams(detector, idle_streams, num_streams);

    double accuracy = (double)idle_count / num_streams * 100.0;

    // Cleanup
    for (int i = 0; i < num_streams; i++) {
        cudaFree(data[i]);
        cudaStreamDestroy(streams[i]);
    }
    free(streams);
    free(data);
    free(idle_streams);
    uvm_idle_detector_shutdown(detector);

    result.value = accuracy;
    result.success = accuracy >= 95.0;

    return result;
}

/*=============================================================================
 * IDLE-004: Thread Safety
 * Measures: Correctness under concurrent updates from multiple threads
 * Target: 100% correctness
 *=============================================================================*/

struct thread_test_data {
    uvm_idle_detector_t *detector;
    int thread_id;
    int launches_per_thread;
    int *error_count;
};

void* thread_launch_worker(void *arg) {
    struct thread_test_data *data = (struct thread_test_data*)arg;

    cudaStream_t stream;
    float *gpu_data;
    cudaStreamCreate(&stream);
    cudaMalloc(&gpu_data, 1024 * 1024 * sizeof(float));

    for (int i = 0; i < data->launches_per_thread; i++) {
        short_compute_kernel<<<128, 128, 0, stream>>>(gpu_data, 100);
        uvm_idle_detector_record_launch(data->detector, stream);
        usleep(5000); // 5ms between launches
    }

    cudaStreamSynchronize(stream);
    cudaFree(gpu_data);
    cudaStreamDestroy(stream);

    return NULL;
}

metric_result_t benchmark_idle_thread_safety(benchmark_config_t *config) {
    metric_result_t result = {0};
    strcpy(result.metric_id, "IDLE-004");
    strcpy(result.name, "Thread Safety");
    strcpy(result.unit, "%");

    const int num_threads = 8;
    const int launches_per_thread = 50;

    uvm_idle_detector_t *detector = uvm_idle_detector_init(0, 100);
    if (!detector) {
        result.success = false;
        strncpy(result.error_message, "Failed to initialize idle detector", sizeof(result.error_message) - 1);
        return result;
    }

    pthread_t *threads = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
    struct thread_test_data *thread_data = (struct thread_test_data*)malloc(num_threads * sizeof(struct thread_test_data));
    int error_count = 0;

    if (!threads || !thread_data) {
        result.success = false;
        strncpy(result.error_message, "Failed to allocate host memory", sizeof(result.error_message) - 1);
        uvm_idle_detector_shutdown(detector);
        return result;
    }

    // Launch threads
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].detector = detector;
        thread_data[i].thread_id = i;
        thread_data[i].launches_per_thread = launches_per_thread;
        thread_data[i].error_count = &error_count;
        pthread_create(&threads[i], NULL, thread_launch_worker, &thread_data[i]);
    }

    // Wait for threads
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    // Check statistics
    uvm_idle_detector_stats_t stats;
    uvm_idle_detector_get_stats(detector, &stats);

    uint64_t expected_launches = num_threads * launches_per_thread;
    double accuracy = (double)stats.total_launches / expected_launches * 100.0;

    free(threads);
    free(thread_data);
    uvm_idle_detector_shutdown(detector);

    result.value = accuracy;
    result.success = (accuracy >= 99.0 && accuracy <= 101.0); // Allow 1% variance

    return result;
}

/*=============================================================================
 * IDLE-005: Performance Overhead
 * Measures: CPU overhead of idle detection per 1000 launches
 * Target: < 10 microseconds per launch
 *=============================================================================*/

metric_result_t benchmark_idle_performance_overhead(benchmark_config_t *config) {
    metric_result_t result = {0};
    strcpy(result.metric_id, "IDLE-005");
    strcpy(result.name, "Performance Overhead");
    strcpy(result.unit, "us/launch");

    const int num_launches = 10000;

    uvm_idle_detector_t *detector = uvm_idle_detector_init(0, 100);
    if (!detector) {
        result.success = false;
        strncpy(result.error_message, "Failed to initialize idle detector", sizeof(result.error_message) - 1);
        return result;
    }

    cudaStream_t stream;
    float *data;
    cudaStreamCreate(&stream);
    cudaMalloc(&data, 1024 * 1024 * sizeof(float));

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < num_launches; i++) {
        uvm_idle_detector_record_launch(detector, stream);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    uint64_t elapsed_ns = (end.tv_sec - start.tv_sec) * 1000000000ULL +
                          (end.tv_nsec - start.tv_nsec);
    double overhead_us_per_launch = (elapsed_ns / 1000.0) / num_launches;

    cudaFree(data);
    cudaStreamDestroy(stream);
    uvm_idle_detector_shutdown(detector);

    result.value = overhead_us_per_launch;
    result.success = overhead_us_per_launch < 10.0;

    return result;
}

/*=============================================================================
 * Benchmark Registration
 *=============================================================================*/

typedef struct {
    const char *metric_id;
    metric_result_t (*func)(benchmark_config_t*);
} idle_benchmark_entry_t;

static const idle_benchmark_entry_t IDLE_BENCHMARKS[] = {
    {"IDLE-001", benchmark_idle_detection_accuracy},
    {"IDLE-002", benchmark_idle_detection_latency},
    {"IDLE-003", benchmark_idle_multistream_scalability},
    {"IDLE-004", benchmark_idle_thread_safety},
    {"IDLE-005", benchmark_idle_performance_overhead},
};

#define NUM_IDLE_BENCHMARKS (sizeof(IDLE_BENCHMARKS) / sizeof(IDLE_BENCHMARKS[0]))

// Run all idle detector benchmarks
extern "C" int run_idle_detector_benchmarks(benchmark_config_t *config, metric_result_t *results, int max_results) {
    int count = 0;

    printf("\n=== Running Idle Detector Benchmarks ===\n");

    for (size_t i = 0; i < NUM_IDLE_BENCHMARKS && count < max_results; i++) {
        printf("\nRunning %s...\n", IDLE_BENCHMARKS[i].metric_id);

        results[count] = IDLE_BENCHMARKS[i].func(config);
        count++;

        // Print result immediately
        metric_result_t *r = &results[count - 1];
        if (r->success) {
            printf("✓ PASS: %s = %.2f %s\n",
                   r->name, r->value, r->unit);
        } else {
            printf("✗ FAIL: %s\n", r->name);
            if (strlen(r->error_message) > 0) {
                printf("  Error: %s\n", r->error_message);
            }
        }
    }

    printf("\nIdle Detector benchmarks completed: %d/%zu metrics\n", count, NUM_IDLE_BENCHMARKS);

    return count;
}
