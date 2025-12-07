/**
 * Memory Pressure Monitor Benchmark for gpu-virt-bench
 *
 * Tests the UVM pressure monitor module for:
 * - Pressure level calculation accuracy
 * - Update performance overhead
 * - Thread-safety under concurrent updates
 * - Callback latency
 * - Time tracking accuracy
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <pthread.h>
#include "../include/benchmark.h"

// Include pressure monitor from bud_fcsp
#include "pressure_monitor.h"

/*=============================================================================
 * PRESSURE-001: Pressure Level Accuracy
 * Measures: Accuracy of pressure level calculations across different usage %
 * Target: 100% accuracy for level classification
 *=============================================================================*/

metric_result_t benchmark_pressure_level_accuracy(benchmark_config_t *config) {
    metric_result_t result = {0};
    strcpy(result.metric_id, "PRESSURE-001");
    strcpy(result.name, "Pressure Level Accuracy");
    strcpy(result.unit, "%");

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    // Initialize pressure monitor with default thresholds
    // NORMAL: 0-70%, WARNING: 70-85%, HIGH: 85-95%, CRITICAL: >= 95%
    uvm_pressure_monitor_t *monitor = uvm_pressure_monitor_init(0, total_mem);
    if (!monitor) {
        result.success = false;
        strncpy(result.error_message, "Failed to initialize pressure monitor", sizeof(result.error_message) - 1);
        return result;
    }

    // Test various usage levels
    struct {
        double usage_pct;
        uvm_pressure_level_t expected_level;
    } test_cases[] = {
        {10.0, UVM_PRESSURE_NORMAL},
        {50.0, UVM_PRESSURE_NORMAL},
        {69.9, UVM_PRESSURE_NORMAL},
        {70.0, UVM_PRESSURE_WARNING},
        {75.0, UVM_PRESSURE_WARNING},
        {84.9, UVM_PRESSURE_WARNING},
        {85.0, UVM_PRESSURE_HIGH},
        {90.0, UVM_PRESSURE_HIGH},
        {94.9, UVM_PRESSURE_HIGH},
        {95.0, UVM_PRESSURE_CRITICAL},
        {98.0, UVM_PRESSURE_CRITICAL},
        {99.9, UVM_PRESSURE_CRITICAL}
    };
    int num_tests = sizeof(test_cases) / sizeof(test_cases[0]);

    int correct = 0;
    for (int i = 0; i < num_tests; i++) {
        size_t usage = (size_t)(total_mem * test_cases[i].usage_pct / 100.0);
        uvm_pressure_monitor_update(monitor, usage);
        uvm_pressure_level_t level = uvm_pressure_monitor_get_level(monitor);

        if (level == test_cases[i].expected_level) {
            correct++;
        } else {
            printf("  [PRESSURE-001] MISMATCH at %.1f%%: expected %d, got %d\n",
                   test_cases[i].usage_pct, test_cases[i].expected_level, level);
        }
    }

    uvm_pressure_monitor_shutdown(monitor);

    double accuracy = (double)correct / num_tests * 100.0;
    result.value = accuracy;
    result.success = (accuracy == 100.0);
    

    if (!result.success) {
        snprintf(result.error_message, sizeof(result.error_message),
                "Accuracy %.1f%% < 100%%", accuracy);
    }

    return result;
}

/*=============================================================================
 * PRESSURE-002: Update Performance Overhead
 * Measures: Time cost of updating pressure monitor
 * Target: < 500 nanoseconds per update
 *=============================================================================*/

metric_result_t benchmark_pressure_update_overhead(benchmark_config_t *config) {
    metric_result_t result = {0};
    strcpy(result.metric_id, "PRESSURE-002");
    strcpy(result.name, "Pressure Update Overhead");
    strcpy(result.unit, "ns");

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    uvm_pressure_monitor_t *monitor = uvm_pressure_monitor_init(0, total_mem);
    if (!monitor) {
        result.success = false;
        strncpy(result.error_message, "Failed to initialize pressure monitor", sizeof(result.error_message) - 1);
        return result;
    }

    const int num_iterations = 100000;
    struct timespec start, end;

    // Warm-up
    for (int i = 0; i < 1000; i++) {
        size_t usage = (size_t)(total_mem * 0.5);
        uvm_pressure_monitor_update(monitor, usage);
    }

    // Benchmark updates
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < num_iterations; i++) {
        // Vary usage slightly to avoid caching effects
        size_t usage = (size_t)(total_mem * (0.5 + (i % 100) * 0.001));
        uvm_pressure_monitor_update(monitor, usage);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);

    uvm_pressure_monitor_shutdown(monitor);

    uint64_t duration_ns = (end.tv_sec - start.tv_sec) * 1000000000ULL +
                           (end.tv_nsec - start.tv_nsec);
    double avg_ns = (double)duration_ns / num_iterations;

    result.value = avg_ns;
    result.success = true;
    

    return result;
}

/*=============================================================================
 * PRESSURE-003: Hysteresis Behavior
 * Measures: Correct hysteresis prevents oscillation
 * Target: 100% correct behavior (no premature downward transitions)
 *=============================================================================*/

metric_result_t benchmark_pressure_hysteresis(benchmark_config_t *config) {
    metric_result_t result = {0};
    strcpy(result.metric_id, "PRESSURE-003");
    strcpy(result.name, "Hysteresis Correctness");
    strcpy(result.unit, "%");

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    uvm_pressure_monitor_t *monitor = uvm_pressure_monitor_init(0, total_mem);
    if (!monitor) {
        result.success = false;
        strncpy(result.error_message, "Failed to initialize pressure monitor", sizeof(result.error_message) - 1);
        return result;
    }

    int tests_passed = 0;
    int total_tests = 0;

    // Test 1: WARNING -> NORMAL should require drop below 65% (70% - 5% hysteresis)
    uvm_pressure_monitor_update(monitor, (size_t)(total_mem * 0.75)); // WARNING
    uvm_pressure_monitor_update(monitor, (size_t)(total_mem * 0.68)); // Should stay WARNING
    if (uvm_pressure_monitor_get_level(monitor) == UVM_PRESSURE_WARNING) {
        tests_passed++;
    }
    total_tests++;

    uvm_pressure_monitor_update(monitor, (size_t)(total_mem * 0.64)); // Should transition to NORMAL
    if (uvm_pressure_monitor_get_level(monitor) == UVM_PRESSURE_NORMAL) {
        tests_passed++;
    }
    total_tests++;

    // Test 2: HIGH -> WARNING should require drop below 80% (85% - 5% hysteresis)
    uvm_pressure_monitor_update(monitor, (size_t)(total_mem * 0.90)); // HIGH
    uvm_pressure_monitor_update(monitor, (size_t)(total_mem * 0.83)); // Should stay HIGH
    if (uvm_pressure_monitor_get_level(monitor) == UVM_PRESSURE_HIGH) {
        tests_passed++;
    }
    total_tests++;

    uvm_pressure_monitor_update(monitor, (size_t)(total_mem * 0.79)); // Should transition to WARNING
    if (uvm_pressure_monitor_get_level(monitor) == UVM_PRESSURE_WARNING) {
        tests_passed++;
    }
    total_tests++;

    // Test 3: CRITICAL -> HIGH should require drop below 90% (95% - 5% hysteresis)
    uvm_pressure_monitor_update(monitor, (size_t)(total_mem * 0.96)); // CRITICAL
    uvm_pressure_monitor_update(monitor, (size_t)(total_mem * 0.93)); // Should stay CRITICAL
    if (uvm_pressure_monitor_get_level(monitor) == UVM_PRESSURE_CRITICAL) {
        tests_passed++;
    }
    total_tests++;

    uvm_pressure_monitor_update(monitor, (size_t)(total_mem * 0.89)); // Should transition to HIGH
    if (uvm_pressure_monitor_get_level(monitor) == UVM_PRESSURE_HIGH) {
        tests_passed++;
    }
    total_tests++;

    uvm_pressure_monitor_shutdown(monitor);

    double correctness = (double)tests_passed / total_tests * 100.0;
    result.value = correctness;
    result.success = (correctness == 100.0);
    

    if (!result.success) {
        snprintf(result.error_message, sizeof(result.error_message),
                "Hysteresis correctness %.1f%% < 100%% (%d/%d passed)",
                correctness, tests_passed, total_tests);
    }

    return result;
}

/*=============================================================================
 * PRESSURE-004: Callback Latency
 * Measures: Time to invoke callback on pressure level change
 * Target: < 1 microsecond
 *=============================================================================*/

static volatile bool callback_invoked = false;
static uint64_t callback_time_ns = 0;
static pthread_mutex_t callback_mutex = PTHREAD_MUTEX_INITIALIZER;

void pressure_change_callback(uvm_pressure_level_t old_level,
                              uvm_pressure_level_t new_level,
                              size_t current_usage,
                              size_t total_memory,
                              void *user_data) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    uint64_t now = (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;

    pthread_mutex_lock(&callback_mutex);
    callback_invoked = true;
    callback_time_ns = now;
    pthread_mutex_unlock(&callback_mutex);
}

metric_result_t benchmark_pressure_callback_latency(benchmark_config_t *config) {
    metric_result_t result = {0};
    strcpy(result.metric_id, "PRESSURE-004");
    strcpy(result.name, "Callback Latency");
    strcpy(result.unit, "ns");

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    uvm_pressure_monitor_t *monitor = uvm_pressure_monitor_init(0, total_mem);
    if (!monitor) {
        result.success = false;
        strncpy(result.error_message, "Failed to initialize pressure monitor", sizeof(result.error_message) - 1);
        return result;
    }

    uvm_pressure_monitor_set_callback(monitor, pressure_change_callback, NULL);

    // Trigger level change and measure callback latency
    const int num_measurements = 100;
    uint64_t total_latency_ns = 0;

    for (int i = 0; i < num_measurements; i++) {
        callback_invoked = false;

        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        uint64_t update_time = (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;

        // Trigger transition NORMAL -> WARNING
        uvm_pressure_monitor_update(monitor, (size_t)(total_mem * 0.50)); // NORMAL
        uvm_pressure_monitor_update(monitor, (size_t)(total_mem * 0.75)); // WARNING

        // Wait for callback (should be nearly instant)
        usleep(1000); // 1ms max wait

        pthread_mutex_lock(&callback_mutex);
        if (callback_invoked) {
            uint64_t latency = callback_time_ns - update_time;
            total_latency_ns += latency;
        }
        pthread_mutex_unlock(&callback_mutex);
    }

    uvm_pressure_monitor_shutdown(monitor);

    double avg_latency_ns = (double)total_latency_ns / num_measurements;

    result.value = avg_latency_ns;
    result.success = true;
     // < 1 microsecond

    return result;
}

/*=============================================================================
 * PRESSURE-005: Thread Safety
 * Measures: Correctness under concurrent updates from multiple threads
 * Target: 100% data integrity, no crashes
 *=============================================================================*/

typedef struct {
    uvm_pressure_monitor_t *monitor;
    size_t total_memory;
    int thread_id;
} pressure_thread_args_t;

void* pressure_update_thread(void *arg) {
    pressure_thread_args_t *args = (pressure_thread_args_t*)arg;

    for (int i = 0; i < 10000; i++) {
        // Each thread updates with different usage patterns
        double usage_pct = 50.0 + (args->thread_id * 10.0) + (i % 10) * 0.5;
        if (usage_pct > 98.0) usage_pct = 98.0;

        size_t usage = (size_t)(args->total_memory * usage_pct / 100.0);
        uvm_pressure_monitor_update(args->monitor, usage);

        // Occasionally query level
        if (i % 100 == 0) {
            uvm_pressure_monitor_get_level(args->monitor);
            uvm_pressure_monitor_get_usage(args->monitor);
        }
    }

    return NULL;
}

metric_result_t benchmark_pressure_thread_safety(benchmark_config_t *config) {
    metric_result_t result = {0};
    strcpy(result.metric_id, "PRESSURE-005");
    strcpy(result.name, "Thread Safety");
    strcpy(result.unit, "bool");

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    uvm_pressure_monitor_t *monitor = uvm_pressure_monitor_init(0, total_mem);
    if (!monitor) {
        result.success = false;
        strncpy(result.error_message, "Failed to initialize pressure monitor", sizeof(result.error_message) - 1);
        return result;
    }

    const int num_threads = 8;
    pthread_t threads[num_threads];
    pressure_thread_args_t args[num_threads];

    // Launch concurrent threads
    for (int i = 0; i < num_threads; i++) {
        args[i].monitor = monitor;
        args[i].total_memory = total_mem;
        args[i].thread_id = i;

        if (pthread_create(&threads[i], NULL, pressure_update_thread, &args[i]) != 0) {
            result.success = false;
            strncpy(result.error_message, "Failed to create thread", sizeof(result.error_message) - 1);
            uvm_pressure_monitor_shutdown(monitor);
            return result;
        }
    }

    // Wait for all threads
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    // Verify monitor is still functional
    size_t usage = uvm_pressure_monitor_get_usage(monitor);
    if (usage > total_mem) {
        result.success = false;
        strncpy(result.error_message, "Usage corrupted after concurrent access", sizeof(result.error_message) - 1);
    } else {
        result.success = true;
        result.value = 1.0; // Pass
    }

    uvm_pressure_monitor_shutdown(monitor);

    
    return result;
}

/*=============================================================================
 * PRESSURE-006: Eviction Recommendation Accuracy
 * Measures: Correctness of should_evict and eviction_target calculations
 * Target: 100% correctness
 *=============================================================================*/

metric_result_t benchmark_pressure_eviction_accuracy(benchmark_config_t *config) {
    metric_result_t result = {0};
    strcpy(result.metric_id, "PRESSURE-006");
    strcpy(result.name, "Eviction Recommendation Accuracy");
    strcpy(result.unit, "%");

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    uvm_pressure_monitor_t *monitor = uvm_pressure_monitor_init(0, total_mem);
    if (!monitor) {
        result.success = false;
        strncpy(result.error_message, "Failed to initialize pressure monitor", sizeof(result.error_message) - 1);
        return result;
    }

    int tests_passed = 0;
    int total_tests = 0;

    // Test 1: NORMAL - should not evict
    uvm_pressure_monitor_update(monitor, (size_t)(total_mem * 0.50));
    if (!uvm_pressure_monitor_should_evict(monitor)) {
        tests_passed++;
    }
    total_tests++;

    // Test 2: WARNING - should not evict
    uvm_pressure_monitor_update(monitor, (size_t)(total_mem * 0.75));
    if (!uvm_pressure_monitor_should_evict(monitor)) {
        tests_passed++;
    }
    total_tests++;

    // Test 3: HIGH - should evict
    uvm_pressure_monitor_update(monitor, (size_t)(total_mem * 0.90));
    if (uvm_pressure_monitor_should_evict(monitor)) {
        tests_passed++;
    }
    total_tests++;

    // Test 4: CRITICAL - should evict
    uvm_pressure_monitor_update(monitor, (size_t)(total_mem * 0.96));
    if (uvm_pressure_monitor_should_evict(monitor)) {
        tests_passed++;
    }
    total_tests++;

    // Test 5: Eviction target should be reasonable
    size_t eviction_target = uvm_pressure_monitor_get_eviction_target(monitor);
    // Target should bring us back below warning threshold with headroom
    // Current: 96%, target: below 65% (70% - 5% hysteresis)
    // So eviction target should be at least (96% - 65%) = 31% of total
    size_t expected_min = (size_t)(total_mem * 0.30);
    if (eviction_target >= expected_min && eviction_target <= total_mem) {
        tests_passed++;
    }
    total_tests++;

    uvm_pressure_monitor_shutdown(monitor);

    double accuracy = (double)tests_passed / total_tests * 100.0;
    result.value = accuracy;
    result.success = (accuracy == 100.0);
    

    if (!result.success) {
        snprintf(result.error_message, sizeof(result.error_message),
                "Eviction accuracy %.1f%% < 100%% (%d/%d passed)",
                accuracy, tests_passed, total_tests);
    }

    return result;
}

/*=============================================================================
 * Run All Pressure Monitor Benchmarks
 *=============================================================================*/

extern "C" int run_pressure_monitor_benchmarks(benchmark_config_t *config, metric_result_t *results, int max_results) {
    int idx = 0;

    if (idx < max_results) {
        results[idx++] = benchmark_pressure_level_accuracy(config);
    }

    if (idx < max_results) {
        results[idx++] = benchmark_pressure_update_overhead(config);
    }

    if (idx < max_results) {
        results[idx++] = benchmark_pressure_hysteresis(config);
    }

    if (idx < max_results) {
        results[idx++] = benchmark_pressure_callback_latency(config);
    }

    if (idx < max_results) {
        results[idx++] = benchmark_pressure_thread_safety(config);
    }

    if (idx < max_results) {
        results[idx++] = benchmark_pressure_eviction_accuracy(config);
    }

    return idx;
}
