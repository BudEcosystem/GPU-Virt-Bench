/*
 * Memory Fragmentation Metrics (FRAG-001 to FRAG-003)
 *
 * Measures GPU memory fragmentation behavior and its impact on
 * allocation performance. Critical for long-running workloads.
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
#define FRAG_WARMUP_ITERATIONS    3
#define FRAG_TEST_ITERATIONS      20
#define FRAG_MAX_ALLOCATIONS      256
#define FRAG_ALLOC_CYCLE_COUNT    10

/* Allocation tracking structure */
typedef struct {
    void *ptr;
    size_t size;
    int active;
} alloc_entry_t;

/* Get current time in microseconds */
static double get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1000.0;
}

/* Random number generator for allocation patterns */
static unsigned int frag_rand_state = 12345;

static unsigned int frag_rand(void) {
    frag_rand_state = frag_rand_state * 1103515245 + 12345;
    return (frag_rand_state >> 16) & 0x7fff;
}

static void frag_srand(unsigned int seed) {
    frag_rand_state = seed;
}

/*
 * FRAG-001: Fragmentation Index
 *
 * Measures memory fragmentation by comparing largest contiguous block
 * to total free memory after a fragmentation-inducing workload.
 */
static bench_result_t bench_fragmentation_index(bench_config_t *config) {
    bench_result_t result;
    memset(&result, 0, sizeof(result));
    strncpy(result.metric_id, METRIC_FRAG_INDEX, sizeof(result.metric_id) - 1);
    strncpy(result.name, "Fragmentation Index", sizeof(result.name) - 1);
    strncpy(result.unit, "%", sizeof(result.unit) - 1);

    /* Get total memory */
    size_t free_mem, total_mem;
    cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
    if (err != cudaSuccess) {
        result.success = 0;
        snprintf(result.error_message, sizeof(result.error_message),
                 "Failed to get memory info: %s", cudaGetErrorString(err));
        return result;
    }

    /* Use 50% of free memory for fragmentation test */
    size_t test_pool = free_mem / 2;
    size_t min_alloc = 1 * 1024 * 1024;    /* 1 MB min */
    size_t max_alloc = 64 * 1024 * 1024;   /* 64 MB max */

    alloc_entry_t allocations[FRAG_MAX_ALLOCATIONS];
    memset(allocations, 0, sizeof(allocations));

    frag_srand((unsigned int)time(NULL));

    double frag_indices[FRAG_TEST_ITERATIONS];
    double total_frag = 0.0;

    for (int iter = 0; iter < FRAG_TEST_ITERATIONS; iter++) {
        /* Phase 1: Allocate many random-sized blocks */
        int num_allocs = 0;
        size_t total_allocated = 0;

        while (total_allocated < test_pool && num_allocs < FRAG_MAX_ALLOCATIONS) {
            size_t alloc_size = min_alloc + (frag_rand() % (max_alloc - min_alloc));

            if (total_allocated + alloc_size > test_pool) {
                alloc_size = test_pool - total_allocated;
            }

            if (alloc_size < min_alloc) break;

            err = cudaMalloc(&allocations[num_allocs].ptr, alloc_size);
            if (err != cudaSuccess) break;

            allocations[num_allocs].size = alloc_size;
            allocations[num_allocs].active = 1;
            total_allocated += alloc_size;
            num_allocs++;
        }

        /* Phase 2: Free alternating blocks to create fragmentation */
        for (int i = 0; i < num_allocs; i += 2) {
            if (allocations[i].active) {
                cudaFree(allocations[i].ptr);
                allocations[i].active = 0;
            }
        }

        /* Phase 3: Try to find the largest contiguous block */
        size_t largest_block = 0;
        size_t probe_size = test_pool / 2;
        void *probe_ptr = NULL;

        /* Binary search for largest allocatable block */
        while (probe_size >= min_alloc) {
            err = cudaMalloc(&probe_ptr, probe_size);
            if (err == cudaSuccess) {
                if (probe_size > largest_block) {
                    largest_block = probe_size;
                }
                cudaFree(probe_ptr);
                probe_size = probe_size + probe_size / 2;
            } else {
                probe_size = probe_size / 2;
            }

            /* Prevent infinite loop */
            if (largest_block > 0 && probe_size > largest_block * 2) break;
        }

        /* Calculate fragmentation index */
        /* Get current free memory */
        size_t current_free, current_total;
        cudaMemGetInfo(&current_free, &current_total);

        /* Fragmentation = 1 - (largest_block / free_memory) * 100 */
        double frag_index = 0.0;
        if (current_free > 0 && largest_block > 0) {
            frag_index = (1.0 - ((double)largest_block / (double)current_free)) * 100.0;
            if (frag_index < 0) frag_index = 0.0;
            if (frag_index > 100.0) frag_index = 100.0;
        }

        frag_indices[iter] = frag_index;
        total_frag += frag_index;

        /* Cleanup remaining allocations */
        for (int i = 0; i < num_allocs; i++) {
            if (allocations[i].active) {
                cudaFree(allocations[i].ptr);
                allocations[i].active = 0;
            }
        }
    }

    result.value = total_frag / FRAG_TEST_ITERATIONS;

    /* Calculate stddev */
    double variance = 0.0;
    for (int i = 0; i < FRAG_TEST_ITERATIONS; i++) {
        double diff = frag_indices[i] - result.value;
        variance += diff * diff;
    }
    result.stddev = sqrt(variance / FRAG_TEST_ITERATIONS);

    result.success = 1;
    result.iterations = FRAG_TEST_ITERATIONS;
    snprintf(result.details, sizeof(result.details),
             "Fragmentation after alloc/free cycles, pool=%zu MB, index=%.1f%%",
             test_pool / (1024 * 1024), result.value);

    return result;
}

/*
 * FRAG-002: Allocation Latency Degradation
 *
 * Measures how allocation latency increases as memory becomes fragmented.
 */
static bench_result_t bench_alloc_latency_degradation(bench_config_t *config) {
    bench_result_t result;
    memset(&result, 0, sizeof(result));
    strncpy(result.metric_id, METRIC_FRAG_ALLOC_LATENCY, sizeof(result.metric_id) - 1);
    strncpy(result.name, "Allocation Latency Degradation", sizeof(result.name) - 1);
    strncpy(result.unit, "x", sizeof(result.unit) - 1);

    size_t test_size = 16 * 1024 * 1024;  /* 16 MB test allocations */

    /* Measure baseline allocation latency on clean heap */
    cudaDeviceSynchronize();

    double baseline_times[20];
    double baseline_total = 0.0;

    for (int i = 0; i < 20; i++) {
        void *ptr;
        double t1 = get_time_us();
        cudaError_t err = cudaMalloc(&ptr, test_size);
        double t2 = get_time_us();

        if (err != cudaSuccess) {
            result.success = 0;
            snprintf(result.error_message, sizeof(result.error_message),
                     "Baseline allocation failed");
            return result;
        }

        baseline_times[i] = t2 - t1;
        baseline_total += baseline_times[i];
        cudaFree(ptr);
    }

    double baseline_avg = baseline_total / 20;

    /* Create fragmentation */
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    size_t fragment_pool = free_mem / 2;
    size_t small_alloc = 2 * 1024 * 1024;  /* 2 MB fragments */

    int max_frags = fragment_pool / small_alloc;
    if (max_frags > FRAG_MAX_ALLOCATIONS) max_frags = FRAG_MAX_ALLOCATIONS;

    void **frag_ptrs = (void**)malloc(max_frags * sizeof(void*));
    int num_frags = 0;

    /* Allocate many small blocks */
    for (int i = 0; i < max_frags; i++) {
        cudaError_t err = cudaMalloc(&frag_ptrs[i], small_alloc);
        if (err != cudaSuccess) break;
        num_frags++;
    }

    /* Free every other block to fragment */
    for (int i = 0; i < num_frags; i += 2) {
        cudaFree(frag_ptrs[i]);
        frag_ptrs[i] = NULL;
    }

    /* Measure allocation latency on fragmented heap */
    double fragmented_times[20];
    double fragmented_total = 0.0;
    int valid_frag_measurements = 0;

    for (int i = 0; i < 20; i++) {
        void *ptr;
        double t1 = get_time_us();
        cudaError_t err = cudaMalloc(&ptr, test_size);
        double t2 = get_time_us();

        if (err == cudaSuccess) {
            fragmented_times[valid_frag_measurements] = t2 - t1;
            fragmented_total += fragmented_times[valid_frag_measurements];
            valid_frag_measurements++;
            cudaFree(ptr);
        }
    }

    /* Cleanup */
    for (int i = 0; i < num_frags; i++) {
        if (frag_ptrs[i] != NULL) {
            cudaFree(frag_ptrs[i]);
        }
    }
    free(frag_ptrs);

    if (valid_frag_measurements == 0) {
        result.success = 0;
        snprintf(result.error_message, sizeof(result.error_message),
                 "No valid fragmented allocations possible");
        return result;
    }

    double fragmented_avg = fragmented_total / valid_frag_measurements;

    /* Calculate degradation factor */
    result.value = fragmented_avg / baseline_avg;

    /* Calculate stddev (based on fragmented measurements variance) */
    double variance = 0.0;
    for (int i = 0; i < valid_frag_measurements; i++) {
        double ratio = fragmented_times[i] / baseline_avg;
        double diff = ratio - result.value;
        variance += diff * diff;
    }
    result.stddev = sqrt(variance / valid_frag_measurements);

    result.success = 1;
    result.iterations = valid_frag_measurements;
    snprintf(result.details, sizeof(result.details),
             "Baseline=%.1f us, Fragmented=%.1f us, Degradation=%.2fx",
             baseline_avg, fragmented_avg, result.value);

    return result;
}

/*
 * FRAG-003: Memory Compaction Efficiency
 *
 * Measures how well memory can be reclaimed after defragmentation.
 * Higher efficiency means better memory management.
 */
static bench_result_t bench_compaction_efficiency(bench_config_t *config) {
    bench_result_t result;
    memset(&result, 0, sizeof(result));
    strncpy(result.metric_id, METRIC_FRAG_COMPACTION, sizeof(result.metric_id) - 1);
    strncpy(result.name, "Memory Compaction Efficiency", sizeof(result.name) - 1);
    strncpy(result.unit, "%", sizeof(result.unit) - 1);

    /* Get initial memory state */
    size_t initial_free, total_mem;
    cudaMemGetInfo(&initial_free, &total_mem);

    size_t test_pool = initial_free / 2;
    size_t block_size = 8 * 1024 * 1024;  /* 8 MB blocks */
    int max_blocks = test_pool / block_size;
    if (max_blocks > FRAG_MAX_ALLOCATIONS) max_blocks = FRAG_MAX_ALLOCATIONS;

    double efficiencies[FRAG_TEST_ITERATIONS];
    double total_efficiency = 0.0;

    for (int iter = 0; iter < FRAG_TEST_ITERATIONS; iter++) {
        /* Get clean state memory */
        size_t clean_free, dummy;
        cudaMemGetInfo(&clean_free, &dummy);

        /* Allocate blocks */
        void **blocks = (void**)malloc(max_blocks * sizeof(void*));
        int num_blocks = 0;

        for (int i = 0; i < max_blocks; i++) {
            cudaError_t err = cudaMalloc(&blocks[i], block_size);
            if (err != cudaSuccess) break;
            num_blocks++;
        }

        if (num_blocks == 0) {
            free(blocks);
            continue;
        }

        /* Free in a pattern that creates fragmentation */
        /* Free blocks 0, 2, 4, 6, ... first */
        for (int i = 0; i < num_blocks; i += 2) {
            cudaFree(blocks[i]);
            blocks[i] = NULL;
        }

        /* Get fragmented memory state */
        size_t frag_free, frag_dummy;
        cudaMemGetInfo(&frag_free, &frag_dummy);

        /* Now free all remaining blocks */
        for (int i = 1; i < num_blocks; i += 2) {
            if (blocks[i] != NULL) {
                cudaFree(blocks[i]);
                blocks[i] = NULL;
            }
        }

        /* Force a device sync to allow any compaction */
        cudaDeviceSynchronize();

        /* Get post-cleanup memory state */
        size_t final_free, final_dummy;
        cudaMemGetInfo(&final_free, &final_dummy);

        /* Calculate compaction efficiency */
        /* Ideal: final_free should be close to clean_free */
        /* Efficiency = (final_free - frag_free) / (clean_free - frag_free) * 100 */
        double efficiency = 100.0;
        if (clean_free > frag_free) {
            double recovered = (double)(final_free - frag_free);
            double expected = (double)(clean_free - frag_free);
            if (expected > 0) {
                efficiency = (recovered / expected) * 100.0;
                if (efficiency > 100.0) efficiency = 100.0;
                if (efficiency < 0.0) efficiency = 0.0;
            }
        }

        efficiencies[iter] = efficiency;
        total_efficiency += efficiency;

        free(blocks);
    }

    result.value = total_efficiency / FRAG_TEST_ITERATIONS;

    /* Calculate stddev */
    double variance = 0.0;
    for (int i = 0; i < FRAG_TEST_ITERATIONS; i++) {
        double diff = efficiencies[i] - result.value;
        variance += diff * diff;
    }
    result.stddev = sqrt(variance / FRAG_TEST_ITERATIONS);

    result.success = 1;
    result.iterations = FRAG_TEST_ITERATIONS;
    snprintf(result.details, sizeof(result.details),
             "Memory recovery after fragmentation, pool=%zu MB, efficiency=%.1f%%",
             test_pool / (1024 * 1024), result.value);

    return result;
}

/*
 * Run all fragmentation metrics
 */
extern "C" void bench_run_fragmentation(bench_config_t *config, bench_result_t *results, int *count) {
    int idx = 0;

    printf("\n=== Memory Fragmentation Metrics ===\n\n");

    /* FRAG-001: Fragmentation Index */
    printf("Running FRAG-001: Fragmentation Index...\n");
    results[idx] = bench_fragmentation_index(config);
    if (results[idx].success) {
        printf("  Result: %.2f %s (+/- %.2f)\n",
               results[idx].value, results[idx].unit, results[idx].stddev);
        printf("  %s\n", results[idx].details);
    } else {
        printf("  FAILED: %s\n", results[idx].error_message);
    }
    idx++;

    /* FRAG-002: Allocation Latency Degradation */
    printf("Running FRAG-002: Allocation Latency Degradation...\n");
    results[idx] = bench_alloc_latency_degradation(config);
    if (results[idx].success) {
        printf("  Result: %.2f %s (+/- %.2f)\n",
               results[idx].value, results[idx].unit, results[idx].stddev);
        printf("  %s\n", results[idx].details);
    } else {
        printf("  FAILED: %s\n", results[idx].error_message);
    }
    idx++;

    /* FRAG-003: Memory Compaction Efficiency */
    printf("Running FRAG-003: Memory Compaction Efficiency...\n");
    results[idx] = bench_compaction_efficiency(config);
    if (results[idx].success) {
        printf("  Result: %.2f %s (+/- %.2f)\n",
               results[idx].value, results[idx].unit, results[idx].stddev);
        printf("  %s\n", results[idx].details);
    } else {
        printf("  FAILED: %s\n", results[idx].error_message);
    }
    idx++;

    *count = idx;
    printf("\nFragmentation metrics completed: %d tests\n", idx);
}
