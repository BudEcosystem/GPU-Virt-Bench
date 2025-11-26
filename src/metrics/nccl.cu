/*
 * NCCL/P2P Communication Metrics (NCCL-001 to NCCL-004)
 *
 * Measures inter-GPU communication performance for GPU virtualization evaluation.
 * These metrics are critical for distributed training workloads.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <cuda_runtime.h>

extern "C" {
#include "include/benchmark.h"
#include "include/metrics.h"
}

/* Configuration */
#define NCCL_WARMUP_ITERATIONS    5
#define NCCL_TEST_ITERATIONS      20
#define NCCL_MIN_SIZE            (1 * 1024 * 1024)      /* 1 MB */
#define NCCL_MAX_SIZE            (256 * 1024 * 1024)    /* 256 MB */

/* GPU copy kernel for P2P simulation */
__global__ void p2p_copy_kernel(float *dst, const float *src, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < n; i += stride) {
        dst[i] = src[i];
    }
}

/* Reduction kernel for AllReduce simulation */
__global__ void reduce_kernel(float *dst, const float *src, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < n; i += stride) {
        dst[i] += src[i];
    }
}

/* Broadcast kernel */
__global__ void broadcast_kernel(float *dst, const float *src, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < n; i += stride) {
        dst[i] = src[i];
    }
}

/*
 * Check if P2P access is available between GPUs
 */
static int check_p2p_available(int gpu0, int gpu1) {
    int can_access_peer = 0;
    cudaError_t err;

    err = cudaDeviceCanAccessPeer(&can_access_peer, gpu0, gpu1);
    if (err != cudaSuccess) {
        return 0;
    }

    return can_access_peer;
}

/*
 * Enable P2P access between GPUs
 */
static int enable_p2p_access(int gpu0, int gpu1) {
    cudaError_t err;

    err = cudaSetDevice(gpu0);
    if (err != cudaSuccess) return -1;

    err = cudaDeviceEnablePeerAccess(gpu1, 0);
    if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
        return -1;
    }

    err = cudaSetDevice(gpu1);
    if (err != cudaSuccess) return -1;

    err = cudaDeviceEnablePeerAccess(gpu0, 0);
    if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
        return -1;
    }

    return 0;
}

/*
 * NCCL-001: AllReduce Latency
 *
 * Measures the latency of AllReduce-style operations across GPUs.
 * AllReduce is the most common collective in distributed training.
 */
static bench_result_t bench_allreduce_latency(bench_config_t *config) {
    bench_result_t result;
    memset(&result, 0, sizeof(result));
    strncpy(result.metric_id, METRIC_NCCL_ALLREDUCE_LATENCY, sizeof(result.metric_id) - 1);
    strncpy(result.name, "AllReduce Latency", sizeof(result.name) - 1);
    strncpy(result.unit, "us", sizeof(result.unit) - 1);

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (err != cudaSuccess || device_count < 1) {
        result.success = 0;
        snprintf(result.error_message, sizeof(result.error_message),
                 "No CUDA devices available");
        return result;
    }

    /* For single GPU, simulate AllReduce with local reduction */
    if (device_count == 1) {
        size_t size = 64 * 1024 * 1024;  /* 64 MB typical gradient size */
        size_t num_elements = size / sizeof(float);

        float *d_data, *d_result;
        err = cudaMalloc(&d_data, size);
        if (err != cudaSuccess) {
            result.success = 0;
            snprintf(result.error_message, sizeof(result.error_message),
                     "Failed to allocate device memory");
            return result;
        }

        err = cudaMalloc(&d_result, size);
        if (err != cudaSuccess) {
            cudaFree(d_data);
            result.success = 0;
            snprintf(result.error_message, sizeof(result.error_message),
                     "Failed to allocate result memory");
            return result;
        }

        /* Initialize data */
        cudaMemset(d_data, 0, size);
        cudaMemset(d_result, 0, size);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        int threads = 256;
        int blocks = (num_elements + threads - 1) / threads;
        if (blocks > 65535) blocks = 65535;

        /* Warmup */
        for (int i = 0; i < NCCL_WARMUP_ITERATIONS; i++) {
            reduce_kernel<<<blocks, threads>>>(d_result, d_data, num_elements);
        }
        cudaDeviceSynchronize();

        /* Measure */
        double total_time = 0.0;
        double times[NCCL_TEST_ITERATIONS];

        for (int i = 0; i < NCCL_TEST_ITERATIONS; i++) {
            cudaEventRecord(start);
            reduce_kernel<<<blocks, threads>>>(d_result, d_data, num_elements);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            times[i] = ms * 1000.0;  /* Convert to microseconds */
            total_time += times[i];
        }

        result.value = total_time / NCCL_TEST_ITERATIONS;

        /* Calculate stddev */
        double variance = 0.0;
        for (int i = 0; i < NCCL_TEST_ITERATIONS; i++) {
            double diff = times[i] - result.value;
            variance += diff * diff;
        }
        result.stddev = sqrt(variance / NCCL_TEST_ITERATIONS);

        result.success = 1;
        result.iterations = NCCL_TEST_ITERATIONS;
        snprintf(result.details, sizeof(result.details),
                 "Single GPU AllReduce simulation, size=%zu MB, latency=%.2f us",
                 size / (1024 * 1024), result.value);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_data);
        cudaFree(d_result);

        return result;
    }

    /* Multi-GPU AllReduce */
    int num_gpus = (device_count > 4) ? 4 : device_count;
    size_t size = 64 * 1024 * 1024;  /* 64 MB */
    size_t num_elements = size / sizeof(float);

    float **d_data = (float**)malloc(num_gpus * sizeof(float*));
    float **d_result = (float**)malloc(num_gpus * sizeof(float*));
    cudaStream_t *streams = (cudaStream_t*)malloc(num_gpus * sizeof(cudaStream_t));

    /* Allocate on each GPU */
    for (int g = 0; g < num_gpus; g++) {
        cudaSetDevice(g);
        cudaMalloc(&d_data[g], size);
        cudaMalloc(&d_result[g], size);
        cudaStreamCreate(&streams[g]);
        cudaMemset(d_data[g], 0, size);
        cudaMemset(d_result[g], 0, size);
    }

    /* Enable P2P where possible */
    for (int i = 0; i < num_gpus; i++) {
        for (int j = 0; j < num_gpus; j++) {
            if (i != j && check_p2p_available(i, j)) {
                enable_p2p_access(i, j);
            }
        }
    }

    cudaEvent_t start, stop;
    cudaSetDevice(0);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    /* Warmup - ring AllReduce simulation */
    for (int w = 0; w < NCCL_WARMUP_ITERATIONS; w++) {
        for (int g = 0; g < num_gpus; g++) {
            cudaSetDevice(g);
            int next = (g + 1) % num_gpus;
            reduce_kernel<<<blocks, threads, 0, streams[g]>>>(d_result[next], d_data[g], num_elements);
        }
        for (int g = 0; g < num_gpus; g++) {
            cudaSetDevice(g);
            cudaStreamSynchronize(streams[g]);
        }
    }

    /* Measure */
    double total_time = 0.0;
    double times[NCCL_TEST_ITERATIONS];

    for (int i = 0; i < NCCL_TEST_ITERATIONS; i++) {
        cudaSetDevice(0);
        cudaEventRecord(start, streams[0]);

        /* Ring AllReduce: reduce-scatter + allgather simulation */
        for (int g = 0; g < num_gpus; g++) {
            cudaSetDevice(g);
            int next = (g + 1) % num_gpus;
            reduce_kernel<<<blocks, threads, 0, streams[g]>>>(d_result[next], d_data[g], num_elements);
        }

        for (int g = 0; g < num_gpus; g++) {
            cudaSetDevice(g);
            cudaStreamSynchronize(streams[g]);
        }

        cudaSetDevice(0);
        cudaEventRecord(stop, streams[0]);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        times[i] = ms * 1000.0;  /* microseconds */
        total_time += times[i];
    }

    result.value = total_time / NCCL_TEST_ITERATIONS;

    /* Calculate stddev */
    double variance = 0.0;
    for (int i = 0; i < NCCL_TEST_ITERATIONS; i++) {
        double diff = times[i] - result.value;
        variance += diff * diff;
    }
    result.stddev = sqrt(variance / NCCL_TEST_ITERATIONS);

    result.success = 1;
    result.iterations = NCCL_TEST_ITERATIONS;
    snprintf(result.details, sizeof(result.details),
             "Multi-GPU AllReduce, GPUs=%d, size=%zu MB, latency=%.2f us",
             num_gpus, size / (1024 * 1024), result.value);

    /* Cleanup */
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    for (int g = 0; g < num_gpus; g++) {
        cudaSetDevice(g);
        cudaFree(d_data[g]);
        cudaFree(d_result[g]);
        cudaStreamDestroy(streams[g]);
    }

    free(d_data);
    free(d_result);
    free(streams);

    return result;
}

/*
 * NCCL-002: AllGather Bandwidth
 *
 * Measures bandwidth of AllGather operations where each GPU contributes
 * its data and all GPUs receive the complete dataset.
 */
static bench_result_t bench_allgather_bandwidth(bench_config_t *config) {
    bench_result_t result;
    memset(&result, 0, sizeof(result));
    strncpy(result.metric_id, METRIC_NCCL_ALLGATHER_BW, sizeof(result.metric_id) - 1);
    strncpy(result.name, "AllGather Bandwidth", sizeof(result.name) - 1);
    strncpy(result.unit, "GB/s", sizeof(result.unit) - 1);

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (err != cudaSuccess || device_count < 1) {
        result.success = 0;
        snprintf(result.error_message, sizeof(result.error_message),
                 "No CUDA devices available");
        return result;
    }

    /* Single GPU simulation */
    if (device_count == 1) {
        size_t chunk_size = 16 * 1024 * 1024;  /* 16 MB per "GPU" */
        int simulated_gpus = 4;
        size_t total_size = chunk_size * simulated_gpus;

        float *d_src, *d_dst;
        err = cudaMalloc(&d_src, chunk_size);
        if (err != cudaSuccess) {
            result.success = 0;
            snprintf(result.error_message, sizeof(result.error_message),
                     "Failed to allocate source memory");
            return result;
        }

        err = cudaMalloc(&d_dst, total_size);
        if (err != cudaSuccess) {
            cudaFree(d_src);
            result.success = 0;
            snprintf(result.error_message, sizeof(result.error_message),
                     "Failed to allocate destination memory");
            return result;
        }

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        /* Warmup */
        for (int i = 0; i < NCCL_WARMUP_ITERATIONS; i++) {
            for (int g = 0; g < simulated_gpus; g++) {
                cudaMemcpyAsync(d_dst + g * (chunk_size / sizeof(float)),
                               d_src, chunk_size, cudaMemcpyDeviceToDevice);
            }
        }
        cudaDeviceSynchronize();

        /* Measure */
        double total_time = 0.0;
        double times[NCCL_TEST_ITERATIONS];

        for (int i = 0; i < NCCL_TEST_ITERATIONS; i++) {
            cudaEventRecord(start);

            for (int g = 0; g < simulated_gpus; g++) {
                cudaMemcpyAsync(d_dst + g * (chunk_size / sizeof(float)),
                               d_src, chunk_size, cudaMemcpyDeviceToDevice);
            }

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            times[i] = ms;
            total_time += ms;
        }

        double avg_time_sec = (total_time / NCCL_TEST_ITERATIONS) / 1000.0;
        double total_bytes = (double)total_size;
        result.value = (total_bytes / avg_time_sec) / (1024.0 * 1024.0 * 1024.0);

        /* Calculate stddev */
        double variance = 0.0;
        double avg_time = total_time / NCCL_TEST_ITERATIONS;
        for (int i = 0; i < NCCL_TEST_ITERATIONS; i++) {
            double diff = times[i] - avg_time;
            variance += diff * diff;
        }
        double time_stddev = sqrt(variance / NCCL_TEST_ITERATIONS);
        result.stddev = result.value * (time_stddev / avg_time);

        result.success = 1;
        result.iterations = NCCL_TEST_ITERATIONS;
        snprintf(result.details, sizeof(result.details),
                 "Single GPU AllGather simulation, chunk=%zu MB, bandwidth=%.2f GB/s",
                 chunk_size / (1024 * 1024), result.value);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_src);
        cudaFree(d_dst);

        return result;
    }

    /* Multi-GPU AllGather */
    int num_gpus = (device_count > 4) ? 4 : device_count;
    size_t chunk_size = 16 * 1024 * 1024;  /* 16 MB per GPU */
    size_t total_size = chunk_size * num_gpus;

    float **d_src = (float**)malloc(num_gpus * sizeof(float*));
    float **d_dst = (float**)malloc(num_gpus * sizeof(float*));
    cudaStream_t *streams = (cudaStream_t*)malloc(num_gpus * sizeof(cudaStream_t));

    for (int g = 0; g < num_gpus; g++) {
        cudaSetDevice(g);
        cudaMalloc(&d_src[g], chunk_size);
        cudaMalloc(&d_dst[g], total_size);
        cudaStreamCreate(&streams[g]);
        cudaMemset(d_src[g], g, chunk_size);
    }

    /* Enable P2P */
    for (int i = 0; i < num_gpus; i++) {
        for (int j = 0; j < num_gpus; j++) {
            if (i != j && check_p2p_available(i, j)) {
                enable_p2p_access(i, j);
            }
        }
    }

    cudaEvent_t start, stop;
    cudaSetDevice(0);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /* Warmup */
    for (int w = 0; w < NCCL_WARMUP_ITERATIONS; w++) {
        for (int g = 0; g < num_gpus; g++) {
            cudaSetDevice(g);
            for (int src = 0; src < num_gpus; src++) {
                cudaMemcpyAsync(d_dst[g] + src * (chunk_size / sizeof(float)),
                               d_src[src], chunk_size, cudaMemcpyDeviceToDevice, streams[g]);
            }
        }
        for (int g = 0; g < num_gpus; g++) {
            cudaSetDevice(g);
            cudaStreamSynchronize(streams[g]);
        }
    }

    /* Measure */
    double total_time = 0.0;
    double times[NCCL_TEST_ITERATIONS];

    for (int i = 0; i < NCCL_TEST_ITERATIONS; i++) {
        cudaSetDevice(0);
        cudaEventRecord(start, streams[0]);

        /* Each GPU gathers from all others */
        for (int g = 0; g < num_gpus; g++) {
            cudaSetDevice(g);
            for (int src = 0; src < num_gpus; src++) {
                cudaMemcpyAsync(d_dst[g] + src * (chunk_size / sizeof(float)),
                               d_src[src], chunk_size, cudaMemcpyDeviceToDevice, streams[g]);
            }
        }

        for (int g = 0; g < num_gpus; g++) {
            cudaSetDevice(g);
            cudaStreamSynchronize(streams[g]);
        }

        cudaSetDevice(0);
        cudaEventRecord(stop, streams[0]);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        times[i] = ms;
        total_time += ms;
    }

    double avg_time_sec = (total_time / NCCL_TEST_ITERATIONS) / 1000.0;
    double bytes_moved = (double)chunk_size * num_gpus * num_gpus;  /* Each GPU receives from all */
    result.value = (bytes_moved / avg_time_sec) / (1024.0 * 1024.0 * 1024.0);

    /* Calculate stddev */
    double variance = 0.0;
    double avg_time = total_time / NCCL_TEST_ITERATIONS;
    for (int i = 0; i < NCCL_TEST_ITERATIONS; i++) {
        double diff = times[i] - avg_time;
        variance += diff * diff;
    }
    double time_stddev = sqrt(variance / NCCL_TEST_ITERATIONS);
    result.stddev = result.value * (time_stddev / avg_time);

    result.success = 1;
    result.iterations = NCCL_TEST_ITERATIONS;
    snprintf(result.details, sizeof(result.details),
             "Multi-GPU AllGather, GPUs=%d, chunk=%zu MB, bandwidth=%.2f GB/s",
             num_gpus, chunk_size / (1024 * 1024), result.value);

    /* Cleanup */
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    for (int g = 0; g < num_gpus; g++) {
        cudaSetDevice(g);
        cudaFree(d_src[g]);
        cudaFree(d_dst[g]);
        cudaStreamDestroy(streams[g]);
    }

    free(d_src);
    free(d_dst);
    free(streams);

    return result;
}

/*
 * NCCL-003: P2P GPU Bandwidth
 *
 * Measures direct peer-to-peer bandwidth between GPUs.
 * Critical for understanding NVLink vs PCIe performance.
 */
static bench_result_t bench_p2p_bandwidth(bench_config_t *config) {
    bench_result_t result;
    memset(&result, 0, sizeof(result));
    strncpy(result.metric_id, METRIC_NCCL_P2P_BANDWIDTH, sizeof(result.metric_id) - 1);
    strncpy(result.name, "P2P GPU Bandwidth", sizeof(result.name) - 1);
    strncpy(result.unit, "GB/s", sizeof(result.unit) - 1);

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (err != cudaSuccess || device_count < 1) {
        result.success = 0;
        snprintf(result.error_message, sizeof(result.error_message),
                 "No CUDA devices available");
        return result;
    }

    /* Single GPU - measure D2D bandwidth on same device */
    if (device_count == 1) {
        size_t sizes[] = {1*1024*1024, 16*1024*1024, 64*1024*1024, 256*1024*1024};
        int num_sizes = 4;
        double max_bandwidth = 0.0;
        size_t best_size = 0;

        for (int s = 0; s < num_sizes; s++) {
            size_t size = sizes[s];

            float *d_src, *d_dst;
            err = cudaMalloc(&d_src, size);
            if (err != cudaSuccess) continue;

            err = cudaMalloc(&d_dst, size);
            if (err != cudaSuccess) {
                cudaFree(d_src);
                continue;
            }

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            /* Warmup */
            for (int i = 0; i < NCCL_WARMUP_ITERATIONS; i++) {
                cudaMemcpy(d_dst, d_src, size, cudaMemcpyDeviceToDevice);
            }

            /* Measure */
            cudaEventRecord(start);
            for (int i = 0; i < NCCL_TEST_ITERATIONS; i++) {
                cudaMemcpy(d_dst, d_src, size, cudaMemcpyDeviceToDevice);
            }
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float ms;
            cudaEventElapsedTime(&ms, start, stop);

            double time_sec = ms / 1000.0;
            double bytes = (double)size * NCCL_TEST_ITERATIONS;
            double bandwidth = (bytes / time_sec) / (1024.0 * 1024.0 * 1024.0);

            if (bandwidth > max_bandwidth) {
                max_bandwidth = bandwidth;
                best_size = size;
            }

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cudaFree(d_src);
            cudaFree(d_dst);
        }

        result.value = max_bandwidth;
        result.stddev = 0.0;  /* Single measurement */
        result.success = 1;
        result.iterations = NCCL_TEST_ITERATIONS;
        snprintf(result.details, sizeof(result.details),
                 "Single GPU D2D bandwidth, best_size=%zu MB, bandwidth=%.2f GB/s",
                 best_size / (1024 * 1024), result.value);

        return result;
    }

    /* Multi-GPU P2P bandwidth measurement */
    double total_bandwidth = 0.0;
    int p2p_pairs = 0;
    double bandwidths[16];  /* Max 4x4 GPU pairs */

    for (int src = 0; src < device_count && src < 4; src++) {
        for (int dst = 0; dst < device_count && dst < 4; dst++) {
            if (src == dst) continue;

            if (!check_p2p_available(src, dst)) {
                continue;
            }

            enable_p2p_access(src, dst);

            size_t size = 64 * 1024 * 1024;  /* 64 MB */

            float *d_src, *d_dst;
            cudaSetDevice(src);
            err = cudaMalloc(&d_src, size);
            if (err != cudaSuccess) continue;

            cudaSetDevice(dst);
            err = cudaMalloc(&d_dst, size);
            if (err != cudaSuccess) {
                cudaSetDevice(src);
                cudaFree(d_src);
                continue;
            }

            cudaEvent_t start, stop;
            cudaSetDevice(src);
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            /* Warmup */
            for (int i = 0; i < NCCL_WARMUP_ITERATIONS; i++) {
                cudaMemcpyPeer(d_dst, dst, d_src, src, size);
            }
            cudaDeviceSynchronize();

            /* Measure */
            cudaEventRecord(start);
            for (int i = 0; i < NCCL_TEST_ITERATIONS; i++) {
                cudaMemcpyPeer(d_dst, dst, d_src, src, size);
            }
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float ms;
            cudaEventElapsedTime(&ms, start, stop);

            double time_sec = ms / 1000.0;
            double bytes = (double)size * NCCL_TEST_ITERATIONS;
            double bandwidth = (bytes / time_sec) / (1024.0 * 1024.0 * 1024.0);

            bandwidths[p2p_pairs] = bandwidth;
            total_bandwidth += bandwidth;
            p2p_pairs++;

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cudaSetDevice(src);
            cudaFree(d_src);
            cudaSetDevice(dst);
            cudaFree(d_dst);
        }
    }

    if (p2p_pairs == 0) {
        result.success = 0;
        snprintf(result.error_message, sizeof(result.error_message),
                 "No P2P-capable GPU pairs found");
        return result;
    }

    result.value = total_bandwidth / p2p_pairs;  /* Average bandwidth */

    /* Calculate stddev */
    double variance = 0.0;
    for (int i = 0; i < p2p_pairs; i++) {
        double diff = bandwidths[i] - result.value;
        variance += diff * diff;
    }
    result.stddev = sqrt(variance / p2p_pairs);

    result.success = 1;
    result.iterations = NCCL_TEST_ITERATIONS;
    snprintf(result.details, sizeof(result.details),
             "P2P bandwidth, pairs=%d, avg=%.2f GB/s, stddev=%.2f GB/s",
             p2p_pairs, result.value, result.stddev);

    return result;
}

/*
 * NCCL-004: Broadcast Bandwidth
 *
 * Measures bandwidth of broadcast operations from one GPU to all others.
 * Important for parameter server and data parallel training patterns.
 */
static bench_result_t bench_broadcast_bandwidth(bench_config_t *config) {
    bench_result_t result;
    memset(&result, 0, sizeof(result));
    strncpy(result.metric_id, METRIC_NCCL_BROADCAST_BW, sizeof(result.metric_id) - 1);
    strncpy(result.name, "Broadcast Bandwidth", sizeof(result.name) - 1);
    strncpy(result.unit, "GB/s", sizeof(result.unit) - 1);

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (err != cudaSuccess || device_count < 1) {
        result.success = 0;
        snprintf(result.error_message, sizeof(result.error_message),
                 "No CUDA devices available");
        return result;
    }

    /* Single GPU simulation */
    if (device_count == 1) {
        size_t size = 64 * 1024 * 1024;  /* 64 MB */
        int simulated_receivers = 4;
        size_t num_elements = size / sizeof(float);

        float *d_src;
        float **d_dst = (float**)malloc(simulated_receivers * sizeof(float*));

        err = cudaMalloc(&d_src, size);
        if (err != cudaSuccess) {
            free(d_dst);
            result.success = 0;
            snprintf(result.error_message, sizeof(result.error_message),
                     "Failed to allocate source memory");
            return result;
        }

        for (int i = 0; i < simulated_receivers; i++) {
            err = cudaMalloc(&d_dst[i], size);
            if (err != cudaSuccess) {
                for (int j = 0; j < i; j++) cudaFree(d_dst[j]);
                cudaFree(d_src);
                free(d_dst);
                result.success = 0;
                snprintf(result.error_message, sizeof(result.error_message),
                         "Failed to allocate destination memory");
                return result;
            }
        }

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        int threads = 256;
        int blocks = (num_elements + threads - 1) / threads;
        if (blocks > 65535) blocks = 65535;

        /* Warmup */
        for (int w = 0; w < NCCL_WARMUP_ITERATIONS; w++) {
            for (int i = 0; i < simulated_receivers; i++) {
                broadcast_kernel<<<blocks, threads>>>(d_dst[i], d_src, num_elements);
            }
        }
        cudaDeviceSynchronize();

        /* Measure */
        double total_time = 0.0;
        double times[NCCL_TEST_ITERATIONS];

        for (int iter = 0; iter < NCCL_TEST_ITERATIONS; iter++) {
            cudaEventRecord(start);

            for (int i = 0; i < simulated_receivers; i++) {
                broadcast_kernel<<<blocks, threads>>>(d_dst[i], d_src, num_elements);
            }

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            times[iter] = ms;
            total_time += ms;
        }

        double avg_time_sec = (total_time / NCCL_TEST_ITERATIONS) / 1000.0;
        double bytes_moved = (double)size * simulated_receivers;
        result.value = (bytes_moved / avg_time_sec) / (1024.0 * 1024.0 * 1024.0);

        /* Calculate stddev */
        double variance = 0.0;
        double avg_time = total_time / NCCL_TEST_ITERATIONS;
        for (int i = 0; i < NCCL_TEST_ITERATIONS; i++) {
            double diff = times[i] - avg_time;
            variance += diff * diff;
        }
        double time_stddev = sqrt(variance / NCCL_TEST_ITERATIONS);
        result.stddev = result.value * (time_stddev / avg_time);

        result.success = 1;
        result.iterations = NCCL_TEST_ITERATIONS;
        snprintf(result.details, sizeof(result.details),
                 "Single GPU broadcast simulation, receivers=%d, size=%zu MB, bandwidth=%.2f GB/s",
                 simulated_receivers, size / (1024 * 1024), result.value);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_src);
        for (int i = 0; i < simulated_receivers; i++) cudaFree(d_dst[i]);
        free(d_dst);

        return result;
    }

    /* Multi-GPU Broadcast */
    int num_gpus = (device_count > 4) ? 4 : device_count;
    int root = 0;  /* GPU 0 is the broadcaster */
    size_t size = 64 * 1024 * 1024;  /* 64 MB */

    float *d_src;
    float **d_dst = (float**)malloc(num_gpus * sizeof(float*));
    cudaStream_t *streams = (cudaStream_t*)malloc(num_gpus * sizeof(cudaStream_t));

    /* Allocate source on root */
    cudaSetDevice(root);
    cudaMalloc(&d_src, size);
    cudaMemset(d_src, 1, size);

    /* Allocate destinations on all GPUs */
    for (int g = 0; g < num_gpus; g++) {
        cudaSetDevice(g);
        cudaMalloc(&d_dst[g], size);
        cudaStreamCreate(&streams[g]);
    }

    /* Enable P2P */
    for (int g = 0; g < num_gpus; g++) {
        if (g != root && check_p2p_available(root, g)) {
            enable_p2p_access(root, g);
        }
    }

    cudaEvent_t start, stop;
    cudaSetDevice(root);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /* Warmup */
    for (int w = 0; w < NCCL_WARMUP_ITERATIONS; w++) {
        for (int g = 0; g < num_gpus; g++) {
            if (g != root) {
                cudaMemcpyPeerAsync(d_dst[g], g, d_src, root, size, streams[g]);
            }
        }
        for (int g = 0; g < num_gpus; g++) {
            cudaSetDevice(g);
            cudaStreamSynchronize(streams[g]);
        }
    }

    /* Measure */
    double total_time = 0.0;
    double times[NCCL_TEST_ITERATIONS];

    for (int iter = 0; iter < NCCL_TEST_ITERATIONS; iter++) {
        cudaSetDevice(root);
        cudaEventRecord(start, streams[root]);

        /* Broadcast from root to all others */
        for (int g = 0; g < num_gpus; g++) {
            if (g != root) {
                cudaMemcpyPeerAsync(d_dst[g], g, d_src, root, size, streams[g]);
            }
        }

        for (int g = 0; g < num_gpus; g++) {
            cudaSetDevice(g);
            cudaStreamSynchronize(streams[g]);
        }

        cudaSetDevice(root);
        cudaEventRecord(stop, streams[root]);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        times[iter] = ms;
        total_time += ms;
    }

    double avg_time_sec = (total_time / NCCL_TEST_ITERATIONS) / 1000.0;
    double bytes_moved = (double)size * (num_gpus - 1);  /* Root sends to n-1 GPUs */
    result.value = (bytes_moved / avg_time_sec) / (1024.0 * 1024.0 * 1024.0);

    /* Calculate stddev */
    double variance = 0.0;
    double avg_time = total_time / NCCL_TEST_ITERATIONS;
    for (int i = 0; i < NCCL_TEST_ITERATIONS; i++) {
        double diff = times[i] - avg_time;
        variance += diff * diff;
    }
    double time_stddev = sqrt(variance / NCCL_TEST_ITERATIONS);
    result.stddev = result.value * (time_stddev / avg_time);

    result.success = 1;
    result.iterations = NCCL_TEST_ITERATIONS;
    snprintf(result.details, sizeof(result.details),
             "Multi-GPU Broadcast, root=%d, receivers=%d, size=%zu MB, bandwidth=%.2f GB/s",
             root, num_gpus - 1, size / (1024 * 1024), result.value);

    /* Cleanup */
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaSetDevice(root);
    cudaFree(d_src);

    for (int g = 0; g < num_gpus; g++) {
        cudaSetDevice(g);
        cudaFree(d_dst[g]);
        cudaStreamDestroy(streams[g]);
    }

    free(d_dst);
    free(streams);

    return result;
}

/*
 * Run all NCCL/P2P metrics
 */
extern "C" void bench_run_nccl(bench_config_t *config, bench_result_t *results, int *count) {
    int idx = 0;

    printf("\n=== NCCL/P2P Communication Metrics ===\n\n");

    /* NCCL-001: AllReduce Latency */
    printf("Running NCCL-001: AllReduce Latency...\n");
    results[idx] = bench_allreduce_latency(config);
    if (results[idx].success) {
        printf("  Result: %.2f %s (+/- %.2f)\n",
               results[idx].value, results[idx].unit, results[idx].stddev);
        printf("  %s\n", results[idx].details);
    } else {
        printf("  FAILED: %s\n", results[idx].error_message);
    }
    idx++;

    /* NCCL-002: AllGather Bandwidth */
    printf("Running NCCL-002: AllGather Bandwidth...\n");
    results[idx] = bench_allgather_bandwidth(config);
    if (results[idx].success) {
        printf("  Result: %.2f %s (+/- %.2f)\n",
               results[idx].value, results[idx].unit, results[idx].stddev);
        printf("  %s\n", results[idx].details);
    } else {
        printf("  FAILED: %s\n", results[idx].error_message);
    }
    idx++;

    /* NCCL-003: P2P GPU Bandwidth */
    printf("Running NCCL-003: P2P GPU Bandwidth...\n");
    results[idx] = bench_p2p_bandwidth(config);
    if (results[idx].success) {
        printf("  Result: %.2f %s (+/- %.2f)\n",
               results[idx].value, results[idx].unit, results[idx].stddev);
        printf("  %s\n", results[idx].details);
    } else {
        printf("  FAILED: %s\n", results[idx].error_message);
    }
    idx++;

    /* NCCL-004: Broadcast Bandwidth */
    printf("Running NCCL-004: Broadcast Bandwidth...\n");
    results[idx] = bench_broadcast_bandwidth(config);
    if (results[idx].success) {
        printf("  Result: %.2f %s (+/- %.2f)\n",
               results[idx].value, results[idx].unit, results[idx].stddev);
        printf("  %s\n", results[idx].details);
    } else {
        printf("  FAILED: %s\n", results[idx].error_message);
    }
    idx++;

    *count = idx;
    printf("\nNCCL/P2P metrics completed: %d tests\n", idx);
}
