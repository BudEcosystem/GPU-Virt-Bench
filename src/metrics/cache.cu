/*
 * Cache Isolation Metrics (CACHE-001 to CACHE-004)
 *
 * Measures L2 cache behavior and isolation under GPU virtualization.
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
#define CACHE_WARMUP_ITERATIONS    5
#define CACHE_TEST_ITERATIONS      20

/* Cache-friendly kernel - sequential access */
__global__ void cache_friendly_kernel(float *data, float *output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (size_t i = idx; i < n; i += stride) {
        sum += data[i];
    }

    if (idx == 0) {
        output[0] = sum;
    }
}

/* Cache-unfriendly kernel - strided access to cause cache misses */
__global__ void cache_unfriendly_kernel(float *data, float *output, size_t n, int stride_factor) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t num_threads = blockDim.x * gridDim.x;

    float sum = 0.0f;
    /* Access with large stride to cause cache thrashing */
    for (size_t i = 0; i < n / stride_factor; i++) {
        size_t access_idx = ((idx + i * stride_factor) * 17) % n;  /* Pseudo-random stride */
        sum += data[access_idx];
    }

    if (idx == 0) {
        output[0] = sum;
    }
}

/* Get L2 cache size in bytes */
static size_t get_l2_cache_size(void) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    return prop.l2CacheSize;
}

/*
 * CACHE-001: L2 Cache Hit Rate
 *
 * Measures effective L2 cache hit rate for sequential access patterns.
 */
static bench_result_t bench_cache_hit_rate(bench_config_t *config) {
    bench_result_t result;
    memset(&result, 0, sizeof(result));
    strncpy(result.metric_id, METRIC_CACHE_L2_HIT_RATE, sizeof(result.metric_id) - 1);
    strncpy(result.name, "L2 Cache Hit Rate", sizeof(result.name) - 1);
    strncpy(result.unit, "%", sizeof(result.unit) - 1);

    size_t l2_size = get_l2_cache_size();
    /* Use data size that fits in L2 cache */
    size_t cache_fit_size = l2_size / 2;
    /* Use data size that exceeds L2 cache */
    size_t cache_exceed_size = l2_size * 4;

    size_t num_elements_fit = cache_fit_size / sizeof(float);
    size_t num_elements_exceed = cache_exceed_size / sizeof(float);

    float *d_data_fit, *d_data_exceed, *d_output;

    cudaError_t err = cudaMalloc(&d_data_fit, cache_fit_size);
    if (err != cudaSuccess) {
        result.success = 0;
        snprintf(result.error_message, sizeof(result.error_message),
                 "Failed to allocate cache-fit memory");
        return result;
    }

    err = cudaMalloc(&d_data_exceed, cache_exceed_size);
    if (err != cudaSuccess) {
        cudaFree(d_data_fit);
        result.success = 0;
        snprintf(result.error_message, sizeof(result.error_message),
                 "Failed to allocate cache-exceed memory");
        return result;
    }

    cudaMalloc(&d_output, sizeof(float));
    cudaMemset(d_data_fit, 1, cache_fit_size);
    cudaMemset(d_data_exceed, 1, cache_exceed_size);

    int threads = 256;
    int blocks_fit = (num_elements_fit + threads - 1) / threads;
    int blocks_exceed = (num_elements_exceed + threads - 1) / threads;
    if (blocks_fit > 65535) blocks_fit = 65535;
    if (blocks_exceed > 65535) blocks_exceed = 65535;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /* Warmup */
    for (int w = 0; w < CACHE_WARMUP_ITERATIONS; w++) {
        cache_friendly_kernel<<<blocks_fit, threads>>>(d_data_fit, d_output, num_elements_fit);
    }
    cudaDeviceSynchronize();

    /* Measure cache-fit performance (should have high hit rate) */
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        cache_friendly_kernel<<<blocks_fit, threads>>>(d_data_fit, d_output, num_elements_fit);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float cache_fit_time;
    cudaEventElapsedTime(&cache_fit_time, start, stop);

    /* Measure cache-exceed performance (should have low hit rate) */
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        cache_friendly_kernel<<<blocks_exceed, threads>>>(d_data_exceed, d_output, num_elements_exceed);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float cache_exceed_time;
    cudaEventElapsedTime(&cache_exceed_time, start, stop);

    /* Calculate effective cache hit rate based on timing difference */
    /* Normalize by data size */
    double time_per_byte_fit = cache_fit_time / (double)cache_fit_size;
    double time_per_byte_exceed = cache_exceed_time / (double)cache_exceed_size;

    /* Cache hit rate approximation: faster access = higher hit rate */
    double speedup = time_per_byte_exceed / time_per_byte_fit;
    double hit_rate = (1.0 - (1.0 / speedup)) * 100.0;
    if (hit_rate < 0) hit_rate = 0;
    if (hit_rate > 100) hit_rate = 100;

    result.value = hit_rate;
    result.stddev = 0.0;
    result.success = 1;
    result.iterations = CACHE_TEST_ITERATIONS;
    snprintf(result.details, sizeof(result.details),
             "Estimated L2 hit rate: %.1f%% (L2 size: %zu KB)",
             hit_rate, l2_size / 1024);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data_fit);
    cudaFree(d_data_exceed);
    cudaFree(d_output);

    return result;
}

/*
 * CACHE-002: Cache Eviction Rate
 *
 * Measures cache eviction rate under concurrent access.
 */
static bench_result_t bench_cache_eviction_rate(bench_config_t *config) {
    bench_result_t result;
    memset(&result, 0, sizeof(result));
    strncpy(result.metric_id, METRIC_CACHE_EVICTION_RATE, sizeof(result.metric_id) - 1);
    strncpy(result.name, "Cache Eviction Rate", sizeof(result.name) - 1);
    strncpy(result.unit, "%", sizeof(result.unit) - 1);

    size_t l2_size = get_l2_cache_size();
    size_t data_size = l2_size / 2;  /* Fits in cache */
    size_t num_elements = data_size / sizeof(float);

    float *d_data1, *d_data2, *d_output;
    cudaStream_t stream1, stream2;

    cudaError_t err = cudaMalloc(&d_data1, data_size);
    if (err != cudaSuccess) {
        result.success = 0;
        snprintf(result.error_message, sizeof(result.error_message),
                 "Failed to allocate data1");
        return result;
    }

    cudaMalloc(&d_data2, data_size);
    cudaMalloc(&d_output, sizeof(float) * 2);
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaMemset(d_data1, 1, data_size);
    cudaMemset(d_data2, 2, data_size);

    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    cudaEvent_t start1, stop1, start2, stop2;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);

    /* Warmup data1 in cache */
    for (int w = 0; w < 5; w++) {
        cache_friendly_kernel<<<blocks, threads, 0, stream1>>>(d_data1, d_output, num_elements);
    }
    cudaDeviceSynchronize();

    /* Measure access time to data1 (should be cached) */
    cudaEventRecord(start1, stream1);
    cache_friendly_kernel<<<blocks, threads, 0, stream1>>>(d_data1, d_output, num_elements);
    cudaEventRecord(stop1, stream1);

    /* Access data2 to cause evictions */
    for (int w = 0; w < 5; w++) {
        cache_friendly_kernel<<<blocks, threads, 0, stream2>>>(d_data2, d_output + 1, num_elements);
    }
    cudaStreamSynchronize(stream2);

    /* Measure access time to data1 again (may have been evicted) */
    cudaEventRecord(start2, stream1);
    cache_friendly_kernel<<<blocks, threads, 0, stream1>>>(d_data1, d_output, num_elements);
    cudaEventRecord(stop2, stream1);
    cudaStreamSynchronize(stream1);

    float time_cached, time_after_eviction;
    cudaEventElapsedTime(&time_cached, start1, stop1);
    cudaEventElapsedTime(&time_after_eviction, start2, stop2);

    /* Calculate eviction impact */
    double slowdown = time_after_eviction / time_cached;
    double eviction_rate = ((slowdown - 1.0) / slowdown) * 100.0;
    if (eviction_rate < 0) eviction_rate = 0;
    if (eviction_rate > 100) eviction_rate = 100;

    result.value = eviction_rate;
    result.stddev = 0.0;
    result.success = 1;
    result.iterations = 1;
    snprintf(result.details, sizeof(result.details),
             "Eviction rate: %.1f%% (slowdown: %.2fx)",
             eviction_rate, slowdown);

    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFree(d_data1);
    cudaFree(d_data2);
    cudaFree(d_output);

    return result;
}

/*
 * CACHE-003: Working Set Collision
 *
 * Measures performance impact when working sets collide in cache.
 */
static bench_result_t bench_working_set_collision(bench_config_t *config) {
    bench_result_t result;
    memset(&result, 0, sizeof(result));
    strncpy(result.metric_id, METRIC_CACHE_WORKING_SET, sizeof(result.metric_id) - 1);
    strncpy(result.name, "Working Set Collision Impact", sizeof(result.name) - 1);
    strncpy(result.unit, "%", sizeof(result.unit) - 1);

    size_t l2_size = get_l2_cache_size();
    size_t working_set_size = l2_size / 4;  /* Each working set is 1/4 of L2 */
    size_t num_elements = working_set_size / sizeof(float);
    int num_working_sets = 4;

    float **d_data = (float**)malloc(num_working_sets * sizeof(float*));
    float *d_output;
    cudaStream_t *streams = (cudaStream_t*)malloc(num_working_sets * sizeof(cudaStream_t));

    for (int i = 0; i < num_working_sets; i++) {
        cudaError_t err = cudaMalloc(&d_data[i], working_set_size);
        if (err != cudaSuccess) {
            for (int j = 0; j < i; j++) {
                cudaFree(d_data[j]);
                cudaStreamDestroy(streams[j]);
            }
            free(d_data); free(streams);
            result.success = 0;
            snprintf(result.error_message, sizeof(result.error_message),
                     "Failed to allocate working set %d", i);
            return result;
        }
        cudaStreamCreate(&streams[i]);
        cudaMemset(d_data[i], i, working_set_size);
    }
    cudaMalloc(&d_output, sizeof(float) * num_working_sets);

    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /* Measure single working set performance (baseline) */
    for (int w = 0; w < 5; w++) {
        cache_friendly_kernel<<<blocks, threads>>>(d_data[0], d_output, num_elements);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        cache_friendly_kernel<<<blocks, threads>>>(d_data[0], d_output, num_elements);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float single_time;
    cudaEventElapsedTime(&single_time, start, stop);

    /* Measure with all working sets active (collision) */
    cudaEventRecord(start);
    for (int rep = 0; rep < 10; rep++) {
        for (int i = 0; i < num_working_sets; i++) {
            cache_friendly_kernel<<<blocks, threads, 0, streams[i]>>>(
                d_data[i], d_output + i, num_elements);
        }
    }
    for (int i = 0; i < num_working_sets; i++) {
        cudaStreamSynchronize(streams[i]);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float collision_time;
    cudaEventElapsedTime(&collision_time, start, stop);

    /* Normalize by number of operations */
    double per_set_single = single_time;
    double per_set_collision = collision_time / num_working_sets;

    double degradation = ((per_set_collision - per_set_single) / per_set_single) * 100.0;
    if (degradation < 0) degradation = 0;

    result.value = degradation;
    result.stddev = 0.0;
    result.success = 1;
    result.iterations = 1;
    snprintf(result.details, sizeof(result.details),
             "%.1f%% degradation with %d working sets",
             degradation, num_working_sets);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    for (int i = 0; i < num_working_sets; i++) {
        cudaFree(d_data[i]);
        cudaStreamDestroy(streams[i]);
    }
    free(d_data);
    free(streams);
    cudaFree(d_output);

    return result;
}

/*
 * CACHE-004: Cache Contention Overhead
 *
 * Measures additional latency from cache contention.
 */
static bench_result_t bench_cache_contention(bench_config_t *config) {
    bench_result_t result;
    memset(&result, 0, sizeof(result));
    strncpy(result.metric_id, METRIC_CACHE_CONTENTION, sizeof(result.metric_id) - 1);
    strncpy(result.name, "Cache Contention Overhead", sizeof(result.name) - 1);
    strncpy(result.unit, "%", sizeof(result.unit) - 1);

    size_t l2_size = get_l2_cache_size();
    size_t data_size = l2_size / 2;
    size_t num_elements = data_size / sizeof(float);

    float *d_data, *d_thrash, *d_output;

    cudaError_t err = cudaMalloc(&d_data, data_size);
    if (err != cudaSuccess) {
        result.success = 0;
        snprintf(result.error_message, sizeof(result.error_message),
                 "Failed to allocate data");
        return result;
    }

    cudaMalloc(&d_thrash, data_size);
    cudaMalloc(&d_output, sizeof(float) * 2);
    cudaMemset(d_data, 1, data_size);
    cudaMemset(d_thrash, 2, data_size);

    cudaStream_t main_stream, thrash_stream;
    cudaStreamCreate(&main_stream);
    cudaStreamCreate(&thrash_stream);

    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /* Baseline: no contention */
    for (int w = 0; w < 5; w++) {
        cache_friendly_kernel<<<blocks, threads, 0, main_stream>>>(d_data, d_output, num_elements);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start, main_stream);
    for (int i = 0; i < 20; i++) {
        cache_friendly_kernel<<<blocks, threads, 0, main_stream>>>(d_data, d_output, num_elements);
    }
    cudaEventRecord(stop, main_stream);
    cudaStreamSynchronize(main_stream);

    float baseline_time;
    cudaEventElapsedTime(&baseline_time, start, stop);

    /* With contention: cache thrashing in parallel */
    cudaEventRecord(start, main_stream);
    for (int i = 0; i < 20; i++) {
        /* Thrash cache in parallel */
        cache_unfriendly_kernel<<<blocks, threads, 0, thrash_stream>>>(d_thrash, d_output + 1, num_elements, 64);
        cache_friendly_kernel<<<blocks, threads, 0, main_stream>>>(d_data, d_output, num_elements);
    }
    cudaStreamSynchronize(thrash_stream);
    cudaEventRecord(stop, main_stream);
    cudaStreamSynchronize(main_stream);

    float contention_time;
    cudaEventElapsedTime(&contention_time, start, stop);

    double overhead = ((contention_time - baseline_time) / baseline_time) * 100.0;
    if (overhead < 0) overhead = 0;

    result.value = overhead;
    result.stddev = 0.0;
    result.success = 1;
    result.iterations = 1;
    snprintf(result.details, sizeof(result.details),
             "%.1f%% overhead from cache contention",
             overhead);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(main_stream);
    cudaStreamDestroy(thrash_stream);
    cudaFree(d_data);
    cudaFree(d_thrash);
    cudaFree(d_output);

    return result;
}

/*
 * Run all cache metrics
 */
extern "C" void bench_run_cache(bench_config_t *config, bench_result_t *results, int *count) {
    int idx = 0;

    printf("\n=== Cache Isolation Metrics ===\n\n");

    /* CACHE-001: L2 Cache Hit Rate */
    printf("Running CACHE-001: L2 Cache Hit Rate...\n");
    results[idx] = bench_cache_hit_rate(config);
    if (results[idx].success) {
        printf("  Result: %.2f %s\n",
               results[idx].value, results[idx].unit);
        printf("  %s\n", results[idx].details);
    } else {
        printf("  FAILED: %s\n", results[idx].error_message);
    }
    idx++;

    /* CACHE-002: Cache Eviction Rate */
    printf("Running CACHE-002: Cache Eviction Rate...\n");
    results[idx] = bench_cache_eviction_rate(config);
    if (results[idx].success) {
        printf("  Result: %.2f %s\n",
               results[idx].value, results[idx].unit);
        printf("  %s\n", results[idx].details);
    } else {
        printf("  FAILED: %s\n", results[idx].error_message);
    }
    idx++;

    /* CACHE-003: Working Set Collision */
    printf("Running CACHE-003: Working Set Collision Impact...\n");
    results[idx] = bench_working_set_collision(config);
    if (results[idx].success) {
        printf("  Result: %.2f %s\n",
               results[idx].value, results[idx].unit);
        printf("  %s\n", results[idx].details);
    } else {
        printf("  FAILED: %s\n", results[idx].error_message);
    }
    idx++;

    /* CACHE-004: Cache Contention */
    printf("Running CACHE-004: Cache Contention Overhead...\n");
    results[idx] = bench_cache_contention(config);
    if (results[idx].success) {
        printf("  Result: %.2f %s\n",
               results[idx].value, results[idx].unit);
        printf("  %s\n", results[idx].details);
    } else {
        printf("  FAILED: %s\n", results[idx].error_message);
    }
    idx++;

    *count = idx;
    printf("\nCache metrics completed: %d tests\n", idx);
}
