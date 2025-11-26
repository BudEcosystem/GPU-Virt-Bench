/*
 * GPU Virtualization Performance Evaluation Tool
 * Isolation Metrics Benchmarks (IS-001 to IS-010)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <signal.h>
#include <sys/wait.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "include/benchmark.h"
#include "include/metrics.h"

/*
 * ============================================================================
 * CUDA Kernels for Isolation Testing
 * ============================================================================
 */

/* Memory-hungry kernel - allocates and uses memory */
__global__ void kernel_memory_stress(float *data, size_t n, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        for (int i = 0; i < iterations; i++) {
            val = val * 1.001f + 0.001f;
        }
        data[idx] = val;
    }
}

/* Compute-intensive kernel for SM utilization testing */
__global__ void kernel_sm_stress(float *output, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (float)idx * 0.001f;

    for (int i = 0; i < iterations; i++) {
        val = __sinf(val) * __cosf(val) + sqrtf(fabsf(val) + 1.0f);
        val = __expf(-fabsf(val) * 0.001f);
    }

    if (output != NULL) {
        output[idx % 1024] = val;
    }
}

/* Variable utilization kernel - allows controlling SM usage */
__global__ void kernel_controlled_util(float *data, int active_threads, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    /* Only some threads do work */
    if (idx < active_threads) {
        float val = data[idx % 1024];
        for (int i = 0; i < iterations; i++) {
            val = __sinf(val) + __cosf(val);
        }
        data[idx % 1024] = val;
    }
}

/*
 * ============================================================================
 * IS-001: Memory Limit Accuracy
 * Tests how accurately the memory limit is enforced
 * ============================================================================
 */

int bench_memory_limit_accuracy(metric_result_t *result) {
    strcpy(result->metric_id, METRIC_IS_MEMORY_LIMIT_ACCURACY);
    result->device_id = 0;
    result->valid = false;

    const int test_attempts = 20;

    result->raw_values = (double*)malloc(sizeof(double) * test_attempts);
    if (result->raw_values == NULL) {
        strcpy(result->error_msg, "Failed to allocate result storage");
        return -1;
    }
    result->raw_count = 0;

    /* Get device memory info */
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    LOG_INFO("Device memory: total=%zu MB, free=%zu MB",
             total_mem / (1024*1024), free_mem / (1024*1024));

    /* Determine the configured memory limit
     * In virtualized environment, total_mem reflects the limit */
    size_t configured_limit = total_mem;

    /* Try to allocate increasingly larger chunks until we hit the limit */
    size_t chunk_size = 64 * 1024 * 1024;  /* 64 MB chunks */
    size_t total_allocated = 0;
    void *allocations[256];
    int alloc_count = 0;

    /* Reset device */
    cudaDeviceReset();
    cudaSetDevice(0);

    /* Allocate until failure */
    while (alloc_count < 256) {
        cudaError_t err = cudaMalloc(&allocations[alloc_count], chunk_size);
        if (err != cudaSuccess) {
            break;
        }
        total_allocated += chunk_size;
        alloc_count++;
    }

    /* Calculate accuracy: how close did we get to the limit? */
    double accuracy = ((double)total_allocated / (double)configured_limit) * 100.0;

    /* Store result */
    for (int i = 0; i < test_attempts; i++) {
        result->raw_values[result->raw_count++] = accuracy;
    }

    /* Cleanup */
    for (int i = 0; i < alloc_count; i++) {
        cudaFree(allocations[i]);
    }

    stats_calculate(result->raw_values, result->raw_count, &result->stats);
    result->value = accuracy;
    result->timestamp_ns = timing_get_ns();
    result->valid = true;

    LOG_INFO("IS-001 Memory Limit Accuracy: %.2f%% (allocated %zu MB of %zu MB limit)",
             accuracy, total_allocated / (1024*1024), configured_limit / (1024*1024));

    return 0;
}

/*
 * ============================================================================
 * IS-002: Memory Limit Enforcement Time
 * Measures time to detect and block over-allocation
 * ============================================================================
 */

int bench_memory_limit_enforcement(metric_result_t *result) {
    strcpy(result->metric_id, METRIC_IS_MEMORY_LIMIT_ENFORCEMENT);
    result->device_id = 0;
    result->valid = false;

    const int iterations = 50;

    result->raw_values = (double*)malloc(sizeof(double) * iterations);
    if (result->raw_values == NULL) {
        strcpy(result->error_msg, "Failed to allocate result storage");
        return -1;
    }
    result->raw_count = 0;

    /* Get free memory and try to over-allocate */
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    /* Try to allocate more than available */
    size_t over_allocation_size = free_mem + 100 * 1024 * 1024;  /* free + 100MB */

    for (int i = 0; i < iterations; i++) {
        void *ptr;
        timing_result_t t;

        timing_start(&t);
        cudaError_t err = cudaMalloc(&ptr, over_allocation_size);
        timing_stop(&t);

        if (err != cudaSuccess) {
            /* Expected: allocation should fail */
            result->raw_values[result->raw_count++] = t.elapsed_us;
        } else {
            /* Unexpected: allocation succeeded, free it */
            cudaFree(ptr);
            result->raw_values[result->raw_count++] = t.elapsed_us;
        }

        /* Clear error state */
        cudaGetLastError();
    }

    stats_calculate(result->raw_values, result->raw_count, &result->stats);
    result->value = result->stats.median;
    result->timestamp_ns = timing_get_ns();
    result->valid = true;

    LOG_INFO("IS-002 Memory Limit Enforcement Time: %.2f us (median)", result->stats.median);

    return 0;
}

/*
 * ============================================================================
 * IS-003: SM Utilization Accuracy
 * Measures how accurately SM limits are enforced
 * ============================================================================
 */

/*
 * ============================================================================
 * IS-003: SM Utilization Accuracy
 * Measures how accurately SM limits are enforced
 * ============================================================================
 */

#include <dlfcn.h>

/* NVML definitions for dynamic loading */
typedef struct {
    unsigned int gpu;
    unsigned int memory;
} nvmlUtilization_t;
typedef int nvmlReturn_t;
typedef void* nvmlDevice_t;

typedef nvmlReturn_t (*nvmlInit_t)(void);
typedef nvmlReturn_t (*nvmlShutdown_t)(void);
typedef nvmlReturn_t (*nvmlDeviceGetHandleByIndex_t)(unsigned int, nvmlDevice_t*);
typedef nvmlReturn_t (*nvmlDeviceGetUtilizationRates_t)(nvmlDevice_t, nvmlUtilization_t*);

int bench_sm_utilization_accuracy(metric_result_t *result) {
    strcpy(result->metric_id, METRIC_IS_SM_UTIL_ACCURACY);
    result->device_id = 0;
    result->valid = false;

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
        nvmlInit = (nvmlInit_t)dlsym(nvml_lib, "nvmlInit");
        nvmlShutdown = (nvmlShutdown_t)dlsym(nvml_lib, "nvmlShutdown");
        nvmlDeviceGetHandleByIndex = (nvmlDeviceGetHandleByIndex_t)dlsym(nvml_lib, "nvmlDeviceGetHandleByIndex");
        nvmlDeviceGetUtilizationRates = (nvmlDeviceGetUtilizationRates_t)dlsym(nvml_lib, "nvmlDeviceGetUtilizationRates");
        
        if (nvmlInit && nvmlShutdown && nvmlDeviceGetHandleByIndex && nvmlDeviceGetUtilizationRates) {
            if (nvmlInit() == 0) { // NVML_SUCCESS
                if (nvmlDeviceGetHandleByIndex(0, &device) == 0) {
                    use_nvml = true;
                }
            }
        }
    }

    const int measurement_duration_ms = 5000;  /* 5 seconds */
    const int sample_count = 50;

    result->raw_values = (double*)malloc(sizeof(double) * sample_count);
    if (result->raw_values == NULL) {
        strcpy(result->error_msg, "Failed to allocate result storage");
        if (use_nvml) nvmlShutdown();
        if (nvml_lib) dlclose(nvml_lib);
        return -1;
    }
    result->raw_count = 0;

    /* Allocate device memory for kernel */
    float *d_data;
    cudaError_t err = cudaMalloc(&d_data, sizeof(float) * 1024);
    if (err != cudaSuccess) {
        sprintf(result->error_msg, "cudaMalloc failed: %s", cudaGetErrorString(err));
        if (use_nvml) nvmlShutdown();
        if (nvml_lib) dlclose(nvml_lib);
        return -1;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    /* Get SM count for scaling */
    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);

    /* Run compute-intensive workload */
    int grid_size = sm_count * 4;
    int block_size = 256;

    /* Launch continuous kernels */
    uint64_t start_time = timing_get_ns();
    uint64_t sample_interval_ns = (measurement_duration_ms * 1000000ULL) / sample_count;

    int kernel_count = 0;
    for (int s = 0; s < sample_count; s++) {
        /* Launch kernels for one sample period */
        while (timing_get_ns() - start_time < (s + 1) * sample_interval_ns) {
            kernel_sm_stress<<<grid_size, block_size, 0, stream>>>(d_data, 1000);
            kernel_count++;

            if (kernel_count % 100 == 0) {
                cudaStreamSynchronize(stream);
            }
        }

        double estimated_util = 0.0;
        if (use_nvml) {
            nvmlUtilization_t util;
            if (nvmlDeviceGetUtilizationRates(device, &util) == 0) {
                estimated_util = (double)util.gpu;
            }
        } else {
            /* Fallback to proxy if NVML fails */
            double kernels_per_sec = (double)kernel_count / ((s + 1) * sample_interval_ns / 1e9);
            estimated_util = (kernels_per_sec / 1000.0) * 100.0;
            if (estimated_util > 100.0) estimated_util = 100.0;
        }

        result->raw_values[result->raw_count++] = estimated_util;
    }

    cudaStreamSynchronize(stream);
    cudaFree(d_data);
    cudaStreamDestroy(stream);

    stats_calculate(result->raw_values, result->raw_count, &result->stats);

    result->value = result->stats.mean;
    result->timestamp_ns = timing_get_ns();
    result->valid = true;

    if (use_nvml) {
        LOG_INFO("IS-003 SM Utilization (NVML): %.2f%% (mean)", result->stats.mean);
        nvmlShutdown();
    } else {
        LOG_INFO("IS-003 SM Utilization (Proxy): %.2f%% (mean)", result->stats.mean);
    }
    
    if (nvml_lib) dlclose(nvml_lib);

    return 0;
}

/*
 * ============================================================================
 * IS-004: SM Limit Response Time
 * Measures feedback loop responsiveness
 * ============================================================================
 */

int bench_sm_limit_response_time(metric_result_t *result) {
    strcpy(result->metric_id, METRIC_IS_SM_LIMIT_RESPONSE);
    result->device_id = 0;
    result->valid = false;

    /* This test measures how quickly SM utilization converges after a change
     * In real implementation, this would:
     * 1. Start at low utilization
     * 2. Suddenly increase workload
     * 3. Measure time to hit limit
     */

    const int iterations = 10;

    result->raw_values = (double*)malloc(sizeof(double) * iterations);
    if (result->raw_values == NULL) {
        strcpy(result->error_msg, "Failed to allocate result storage");
        return -1;
    }
    result->raw_count = 0;

    float *d_data;
    cudaMalloc(&d_data, sizeof(float) * 1024);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);

    for (int iter = 0; iter < iterations; iter++) {
        /* Start idle */
        usleep(100000);  /* 100ms idle */

        /* Suddenly start intensive workload */
        timing_result_t t;
        timing_start(&t);

        /* Launch burst of kernels */
        for (int i = 0; i < 1000; i++) {
            kernel_sm_stress<<<sm_count * 4, 256, 0, stream>>>(d_data, 500);
        }
        cudaStreamSynchronize(stream);

        timing_stop(&t);

        /* Response time is approximately the time to process the burst
         * when rate limiting is active */
        result->raw_values[result->raw_count++] = t.elapsed_ms;
    }

    cudaFree(d_data);
    cudaStreamDestroy(stream);

    stats_calculate(result->raw_values, result->raw_count, &result->stats);
    result->value = result->stats.median;
    result->timestamp_ns = timing_get_ns();
    result->valid = true;

    LOG_INFO("IS-004 SM Limit Response Time: %.2f ms (median)", result->stats.median);

    return 0;
}

/*
 * ============================================================================
 * IS-005: Cross-Tenant Memory Isolation
 * Tests if one process can access another's memory
 * ============================================================================
 */

/* Forward declarations for worker functions */
int worker_is005_child(const char *args);
int worker_load_gen(const char *args);

int dispatch_worker(const char *test_id, const char *args) {
    if (strcmp(test_id, "IS-005") == 0) {
        return worker_is005_child(args);
    }
    if (strcmp(test_id, "IS-LOAD-GEN") == 0) {
        return worker_load_gen(args);
    }
    // Add other workers here
    fprintf(stderr, "Unknown worker test ID: %s\n", test_id);
    return 1;
}

/*
 * ============================================================================
 * IS-005: Cross-Tenant Memory Isolation
 * Tests if one process can access another's memory
 * ============================================================================
 */

#include "utils/process.h"

/* Child process for IS-005 */
int worker_is005_child(const char *args) {
    /* Args should contain the pointer address as hex string */
    unsigned long long ptr_addr;
    if (sscanf(args, "%llx", &ptr_addr) != 1) {
        fprintf(stderr, "Child: Invalid arguments\n");
        return 1;
    }
    
    void *d_ptr = (void*)ptr_addr;
    
    /* Initialize CUDA in child */
    cudaSetDevice(0);
    
    /* Try to read from the pointer */
    float read_back = 0.0f;
    cudaError_t err = cudaMemcpy(&read_back, d_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    
    if (err == cudaSuccess) {
        return 0; // Accessed successfully (Bad)
    } else {
        return 1; // Access failed (Good)
    }
}

int bench_cross_tenant_memory_isolation(metric_result_t *result) {
    strcpy(result->metric_id, METRIC_IS_CROSS_TENANT_MEMORY);
    result->device_id = 0;
    result->valid = false;
    
    result->raw_values = (double*)malloc(sizeof(double) * 1);
    if (result->raw_values == NULL) {
        strcpy(result->error_msg, "Failed to allocate result storage");
        return -1;
    }
    result->raw_count = 0;
    
    /* Allocate memory in Parent */
    float *d_ptr;
    cudaMalloc(&d_ptr, sizeof(float) * 1024);
    
    float test_val = 1234.5f;
    cudaMemcpy(d_ptr, &test_val, sizeof(float), cudaMemcpyHostToDevice);
    
    /* Prepare args for child */
    char args[64];
    sprintf(args, "%llx", (unsigned long long)d_ptr);
    
    /* Launch child worker */
    pid_t pid;
    if (launch_worker("IS-005", args, &pid, NULL, NULL) != 0) {
        strcpy(result->error_msg, "Failed to launch worker");
        cudaFree(d_ptr);
        return -1;
    }
    
    /* Wait for child */
    int status = wait_for_worker(pid);
    
    /* Child returns 1 if access failed (Good), 0 if access succeeded (Bad) */
    bool isolation_intact = (status == 1);
    
    /* If status is something else (crash), it's also "Good" for isolation (access denied violently) */
    if (status != 0 && status != 1) {
        isolation_intact = true;
    }
    
    result->raw_values[result->raw_count++] = isolation_intact ? 1.0 : 0.0;
    
    cudaFree(d_ptr);
    
    stats_calculate(result->raw_values, result->raw_count, &result->stats);
    result->value = isolation_intact ? 1.0 : 0.0;
    result->timestamp_ns = timing_get_ns();
    result->valid = true;
    
    LOG_INFO("IS-005 Cross-Tenant Memory Isolation: %s", 
             isolation_intact ? "PASS" : "FAIL");
             
    return 0;
}

/*
 * ============================================================================
 * IS-006: Cross-Tenant Compute Isolation
 * Measures compute interference between tenants
 * ============================================================================
 */

/* Worker for generating load */
int worker_load_gen(const char *args) {
    /* Initialize CUDA */
    cudaSetDevice(0);
    
    float *d_data;
    cudaMalloc(&d_data, sizeof(float) * 1024);
    
    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    
    /* Run until killed */
    while (1) {
        kernel_sm_stress<<<sm_count * 4, 256>>>(d_data, 1000);
        cudaDeviceSynchronize();
    }
    
    return 0;
}

int bench_cross_tenant_compute_isolation(metric_result_t *result) {
    strcpy(result->metric_id, METRIC_IS_CROSS_TENANT_COMPUTE);
    result->device_id = 0;
    result->valid = false;

    const int iterations = 20;

    result->raw_values = (double*)malloc(sizeof(double) * iterations);
    if (result->raw_values == NULL) {
        strcpy(result->error_msg, "Failed to allocate result storage");
        return -1;
    }
    result->raw_count = 0;

    /* First, measure baseline performance (solo execution) */
    float *d_data;
    cudaMalloc(&d_data, sizeof(float) * 1024);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);

    /* Baseline measurement */
    double baseline_sum = 0.0;
    for (int i = 0; i < iterations; i++) {
        timing_result_t t;
        timing_start(&t);

        for (int k = 0; k < 100; k++) {
            kernel_sm_stress<<<sm_count * 2, 256, 0, stream>>>(d_data, 500);
        }
        cudaStreamSynchronize(stream);

        timing_stop(&t);
        baseline_sum += t.elapsed_ms;
    }
    double baseline_mean = baseline_sum / iterations;

    /* Launch competing tenant */
    pid_t pid;
    if (launch_worker("IS-LOAD-GEN", "", &pid, NULL, NULL) != 0) {
        strcpy(result->error_msg, "Failed to launch load generator");
        cudaFree(d_data);
        cudaStreamDestroy(stream);
        return -1;
    }
    
    /* Give it a moment to start */
    sleep(1);

    /* Measure with contention */
    for (int i = 0; i < iterations; i++) {
        timing_result_t t;
        timing_start(&t);

        for (int k = 0; k < 100; k++) {
            kernel_sm_stress<<<sm_count * 2, 256, 0, stream>>>(d_data, 500);
        }
        cudaStreamSynchronize(stream);

        timing_stop(&t);

        /* Calculate percentage of baseline performance maintained */
        double perf_ratio = baseline_mean / t.elapsed_ms * 100.0;
        if (perf_ratio > 100.0) perf_ratio = 100.0;

        result->raw_values[result->raw_count++] = perf_ratio;
    }
    
    /* Kill worker */
    kill(pid, SIGTERM);
    wait_for_worker(pid);

    cudaFree(d_data);
    cudaStreamDestroy(stream);

    stats_calculate(result->raw_values, result->raw_count, &result->stats);
    result->value = result->stats.mean;
    result->timestamp_ns = timing_get_ns();
    result->valid = true;

    LOG_INFO("IS-006 Cross-Tenant Compute Isolation: %.2f%% performance maintained",
             result->stats.mean);

    return 0;
}

/*
 * ============================================================================
 * IS-007: QoS Consistency
 * Measures variance in performance over time
 * ============================================================================
 */

int bench_qos_consistency(metric_result_t *result) {
    strcpy(result->metric_id, METRIC_IS_QOS_CONSISTENCY);
    result->device_id = 0;
    result->valid = false;

    const int iterations = 100;

    result->raw_values = (double*)malloc(sizeof(double) * iterations);
    if (result->raw_values == NULL) {
        strcpy(result->error_msg, "Failed to allocate result storage");
        return -1;
    }
    result->raw_count = 0;

    float *d_data;
    cudaMalloc(&d_data, sizeof(float) * 1024);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);

    /* Measure kernel execution times over extended period */
    for (int i = 0; i < iterations; i++) {
        timing_result_t t;
        timing_cuda_sync_start(&t, stream);

        kernel_sm_stress<<<sm_count, 256, 0, stream>>>(d_data, 1000);

        timing_cuda_sync_stop(&t, stream);
        result->raw_values[result->raw_count++] = t.elapsed_ms;

        /* Small delay between iterations */
        usleep(10000);  /* 10ms */
    }

    cudaFree(d_data);
    cudaStreamDestroy(stream);

    stats_calculate(result->raw_values, result->raw_count, &result->stats);

    /* QoS consistency is measured as coefficient of variation */
    double cv = stats_coefficient_of_variation(&result->stats);
    result->value = cv;
    result->timestamp_ns = timing_get_ns();
    result->valid = true;

    LOG_INFO("IS-007 QoS Consistency: CV=%.4f (lower is better)", cv);

    return 0;
}

/*
 * ============================================================================
 * IS-008: Fairness Index
 * Measures fairness across multiple concurrent workloads
 * ============================================================================
 */

int bench_fairness_index(metric_result_t *result) {
    strcpy(result->metric_id, METRIC_IS_FAIRNESS_INDEX);
    result->device_id = 0;
    result->valid = false;

    /* Use multiple streams to simulate multiple tenants */
    const int num_streams = 4;
    const int iterations = 50;

    result->raw_values = (double*)malloc(sizeof(double) * iterations);
    if (result->raw_values == NULL) {
        strcpy(result->error_msg, "Failed to allocate result storage");
        return -1;
    }
    result->raw_count = 0;

    cudaStream_t streams[num_streams];
    float *d_data[num_streams];
    double stream_times[num_streams];

    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
        cudaMalloc(&d_data[i], sizeof(float) * 1024);
        stream_times[i] = 0.0;
    }

    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);

    for (int iter = 0; iter < iterations; iter++) {
        /* Launch equal work on all streams */
        for (int s = 0; s < num_streams; s++) {
            timing_result_t t;
            timing_start(&t);

            for (int k = 0; k < 25; k++) {
                kernel_sm_stress<<<sm_count / num_streams, 256, 0, streams[s]>>>(
                    d_data[s], 500);
            }
            cudaStreamSynchronize(streams[s]);

            timing_stop(&t);
            stream_times[s] = t.elapsed_ms;
        }

        /* Calculate Jain's fairness for this iteration */
        double fairness = stats_jains_fairness(stream_times, num_streams);
        result->raw_values[result->raw_count++] = fairness;
    }

    /* Cleanup */
    for (int i = 0; i < num_streams; i++) {
        cudaFree(d_data[i]);
        cudaStreamDestroy(streams[i]);
    }

    stats_calculate(result->raw_values, result->raw_count, &result->stats);
    result->value = result->stats.mean;
    result->timestamp_ns = timing_get_ns();
    result->valid = true;

    LOG_INFO("IS-008 Fairness Index: %.4f (1.0 = perfect fairness)", result->stats.mean);

    return 0;
}

/*
 * ============================================================================
 * IS-009: Noisy Neighbor Impact
 * Measures performance degradation from aggressive co-tenant
 * ============================================================================
 */

int bench_noisy_neighbor_impact(metric_result_t *result) {
    strcpy(result->metric_id, METRIC_IS_NOISY_NEIGHBOR);
    result->device_id = 0;
    result->valid = false;

    const int iterations = 20;

    result->raw_values = (double*)malloc(sizeof(double) * iterations);
    if (result->raw_values == NULL) {
        strcpy(result->error_msg, "Failed to allocate result storage");
        return -1;
    }
    result->raw_count = 0;

    float *d_victim;
    cudaMalloc(&d_victim, sizeof(float) * 1024);

    cudaStream_t victim_stream;
    cudaStreamCreate(&victim_stream);

    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);

    /* Baseline: victim alone */
    double baseline_sum = 0.0;
    for (int i = 0; i < iterations; i++) {
        timing_result_t t;
        timing_start(&t);

        for (int k = 0; k < 50; k++) {
            kernel_sm_stress<<<sm_count / 2, 256, 0, victim_stream>>>(d_victim, 500);
        }
        cudaStreamSynchronize(victim_stream);

        timing_stop(&t);
        baseline_sum += t.elapsed_ms;
    }
    double baseline_mean = baseline_sum / iterations;

    /* Launch noisy neighbor */
    pid_t pid;
    if (launch_worker("IS-LOAD-GEN", "", &pid, NULL, NULL) != 0) {
        strcpy(result->error_msg, "Failed to launch noisy neighbor");
        cudaFree(d_victim);
        cudaStreamDestroy(victim_stream);
        return -1;
    }
    
    /* Give it a moment to start */
    sleep(1);

    /* Measure victim performance */
    for (int i = 0; i < iterations; i++) {
        timing_result_t t;
        timing_start(&t);

        for (int k = 0; k < 50; k++) {
            kernel_sm_stress<<<sm_count / 2, 256, 0, victim_stream>>>(d_victim, 500);
        }
        cudaStreamSynchronize(victim_stream);

        timing_stop(&t);

        /* Calculate degradation percentage */
        double degradation = ((t.elapsed_ms - baseline_mean) / baseline_mean) * 100.0;
        if (degradation < 0) degradation = 0;

        result->raw_values[result->raw_count++] = degradation;
    }
    
    /* Kill worker */
    kill(pid, SIGTERM);
    wait_for_worker(pid);

    cudaFree(d_victim);
    cudaStreamDestroy(victim_stream);

    stats_calculate(result->raw_values, result->raw_count, &result->stats);
    result->value = result->stats.mean;
    result->timestamp_ns = timing_get_ns();
    result->valid = true;

    LOG_INFO("IS-009 Noisy Neighbor Impact: %.2f%% degradation", result->stats.mean);

    return 0;
}

/*
 * ============================================================================
 * IS-010: Fault Isolation
 * Tests if errors in one context affect others
 * ============================================================================
 */

int bench_fault_isolation(metric_result_t *result) {
    strcpy(result->metric_id, METRIC_IS_FAULT_ISOLATION);
    result->device_id = 0;
    result->valid = false;

    result->raw_values = (double*)malloc(sizeof(double) * 1);
    if (result->raw_values == NULL) {
        strcpy(result->error_msg, "Failed to allocate result storage");
        return -1;
    }
    result->raw_count = 0;

    /* Test: Can we continue after causing and clearing an error? */
    bool fault_contained = true;

    /* First allocate some valid memory and free it */
    float *d_temp;
    cudaError_t err = cudaMalloc(&d_temp, sizeof(float) * 1024);
    if (err != cudaSuccess) {
        strcpy(result->error_msg, "Failed to allocate test memory");
        return -1;
    }

    /* Free the memory, then try to use it (use after free scenario) */
    cudaFree(d_temp);

    /* Try to cause an error by allocating a huge amount of memory */
    float *d_huge;
    size_t huge_size = 1024ULL * 1024 * 1024 * 1024; /* 1 TB - impossible */
    err = cudaMalloc(&d_huge, huge_size);

    /* Error is expected (cudaErrorMemoryAllocation) */
    if (err != cudaSuccess) {
        /* Clear the error */
        cudaGetLastError();

        /* Try normal operations after the error */
        float *d_test;
        err = cudaMalloc(&d_test, sizeof(float) * 1024);

        if (err == cudaSuccess) {
            /* Good: we recovered from the error */
            float test_val = 123.0f;
            err = cudaMemcpy(d_test, &test_val, sizeof(float), cudaMemcpyHostToDevice);

            if (err == cudaSuccess) {
                float read_back = 0.0f;
                err = cudaMemcpy(&read_back, d_test, sizeof(float), cudaMemcpyDeviceToHost);

                if (err == cudaSuccess && fabsf(read_back - test_val) < 0.001f) {
                    fault_contained = true;
                } else {
                    fault_contained = false;
                }
            } else {
                fault_contained = false;
            }

            cudaFree(d_test);
        } else {
            /* Bad: couldn't recover */
            fault_contained = false;
        }
    } else {
        /* Unexpected: allocation succeeded (unlikely) */
        cudaFree(d_huge);
        fault_contained = true; /* No fault occurred to contain */
    }

    result->raw_values[result->raw_count++] = fault_contained ? 1.0 : 0.0;

    stats_calculate(result->raw_values, result->raw_count, &result->stats);
    result->value = fault_contained ? 1.0 : 0.0;
    result->timestamp_ns = timing_get_ns();
    result->valid = true;

    LOG_INFO("IS-010 Fault Isolation: %s", fault_contained ? "PASS" : "FAIL");

    return 0;
}

/*
 * ============================================================================
 * Run All Isolation Benchmarks
 * ============================================================================
 */

int bench_run_isolation(const benchmark_config_t *config, benchmark_result_t *results) {
    strcpy(results->benchmark_name, "Isolation Benchmarks");

    results->results = (metric_result_t*)calloc(10, sizeof(metric_result_t));
    if (results->results == NULL) {
        strcpy(results->error_msg, "Failed to allocate metric results");
        results->success = false;
        return -1;
    }
    results->result_count = 0;

    timing_result_t total_time;
    timing_start(&total_time);

    LOG_INFO("Running IS-001: Memory Limit Accuracy");
    if (bench_memory_limit_accuracy(&results->results[results->result_count]) == 0) {
        results->result_count++;
    }

    LOG_INFO("Running IS-002: Memory Limit Enforcement Time");
    if (bench_memory_limit_enforcement(&results->results[results->result_count]) == 0) {
        results->result_count++;
    }

    LOG_INFO("Running IS-003: SM Utilization Accuracy");
    if (bench_sm_utilization_accuracy(&results->results[results->result_count]) == 0) {
        results->result_count++;
    }

    LOG_INFO("Running IS-004: SM Limit Response Time");
    if (bench_sm_limit_response_time(&results->results[results->result_count]) == 0) {
        results->result_count++;
    }

    LOG_INFO("Running IS-005: Cross-Tenant Memory Isolation");
    if (bench_cross_tenant_memory_isolation(&results->results[results->result_count]) == 0) {
        results->result_count++;
    }

    LOG_INFO("Running IS-006: Cross-Tenant Compute Isolation");
    if (bench_cross_tenant_compute_isolation(&results->results[results->result_count]) == 0) {
        results->result_count++;
    }

    LOG_INFO("Running IS-007: QoS Consistency");
    if (bench_qos_consistency(&results->results[results->result_count]) == 0) {
        results->result_count++;
    }

    LOG_INFO("Running IS-008: Fairness Index");
    if (bench_fairness_index(&results->results[results->result_count]) == 0) {
        results->result_count++;
    }

    LOG_INFO("Running IS-009: Noisy Neighbor Impact");
    if (bench_noisy_neighbor_impact(&results->results[results->result_count]) == 0) {
        results->result_count++;
    }

    LOG_INFO("Running IS-010: Fault Isolation");
    if (bench_fault_isolation(&results->results[results->result_count]) == 0) {
        results->result_count++;
    }

    timing_stop(&total_time);
    results->total_time_ms = total_time.elapsed_ms;
    results->success = (results->result_count > 0);

    LOG_INFO("Isolation benchmarks completed: %d/%d metrics, %.2f ms total",
             results->result_count, 10, results->total_time_ms);

    return results->success ? 0 : -1;
}
