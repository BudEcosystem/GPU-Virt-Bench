/*
 * GPU Virtualization Performance Evaluation Tool
 * FCSP Advanced Feature Metrics (FCSP-001 to FCSP-012)
 *
 * Metrics for testing FCSP's new features:
 * - Workload Class Affinity Scheduling
 * - Adaptive SM Limiting via Utilization Monitoring
 * - UVM-Based Memory Orchestration
 * - Prefetch on Stream Activation
 *
 * These metrics validate the features work correctly and measure their impact.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "include/benchmark.h"
#include "include/metrics.h"

/*
 * ============================================================================
 * CUDA Kernels for Workload Classification Testing
 * ============================================================================
 */

/* Compute-intensive kernel: heavy FLOPS, minimal memory */
__global__ void fcsp_kernel_compute_heavy(float *scratch, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (float)(idx + 1);

    /* Heavy transcendental compute */
    #pragma unroll 4
    for (int i = 0; i < iterations; i++) {
        val = sinf(val) * cosf(val) + sqrtf(fabsf(val));
        val = expf(-val * 0.001f) + tanf(val * 0.01f);
        val = logf(fabsf(val) + 1.0f) * powf(val, 0.5f);
    }

    /* Minimal memory write */
    if (scratch != NULL && idx == 0) {
        scratch[0] = val;
    }
}

/* Memory-intensive kernel: high bandwidth, minimal compute */
__global__ void fcsp_kernel_memory_heavy(float *src, float *dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    /* Stream through memory with minimal compute */
    for (int i = idx; i < n; i += stride) {
        dst[i] = src[i];
    }
}

/* Balanced kernel: moderate compute and memory */
__global__ void fcsp_kernel_balanced(float *data, int n, int compute_iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float val = data[idx];
        for (int i = 0; i < compute_iters; i++) {
            val = sqrtf(val + 1.0f) * 0.99f;
        }
        data[idx] = val;
    }
}

/* Attention-like kernel: small grid, large block */
__global__ void fcsp_kernel_attention_pattern(float *qkv, float *out,
                                               int seq_len, int head_dim) {
    int head_idx = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float shared[];

    /* Simulate attention pattern: load Q, K, V, compute softmax(QK^T)V */
    float score = 0.0f;
    for (int i = 0; i < seq_len; i++) {
        float q = qkv[head_idx * head_dim + tid];
        float k = qkv[head_idx * head_dim + (tid + i) % head_dim];
        score += q * k;
    }

    shared[tid] = expf(score * 0.125f);  /* Scale by 1/sqrt(d_k) */
    __syncthreads();

    /* Simple reduction for demonstration */
    if (tid == 0) {
        float sum = 0;
        for (int i = 0; i < blockDim.x && i < head_dim; i++) {
            sum += shared[i];
        }
        out[head_idx] = sum;
    }
}

/* FFN-like kernel: large grid, standard block */
__global__ void fcsp_kernel_ffn_pattern(float *input, float *weight,
                                         float *output, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += input[row * n + k] * weight[k * n + col];
        }
        /* GELU activation approximation */
        float gelu = sum * 0.5f * (1.0f + tanhf(0.7978845608f *
                     (sum + 0.044715f * sum * sum * sum)));
        output[row * n + col] = gelu;
    }
}

/*
 * ============================================================================
 * FCSP-001: Workload Affinity - Compute+Memory Concurrency Gain
 * Tests if compute-heavy and memory-heavy workloads run efficiently together
 * ============================================================================
 */

int bench_fcsp_affinity_complementary(metric_result_t *result) {
    const int iterations = 50;
    const int warmup = 10;
    const size_t mem_size = 64 * 1024 * 1024;  /* 64MB for memory kernel */

    strcpy(result->metric_id, "FCSP-001");
    result->device_id = 0;
    result->valid = false;

    result->raw_values = (double*)malloc(sizeof(double) * iterations);
    if (!result->raw_values) {
        strcpy(result->error_msg, "Failed to allocate result storage");
        return -1;
    }
    result->raw_count = 0;

    /* Declare all variables before any conditional returns */
    float *d_scratch = NULL, *d_src = NULL, *d_dst = NULL;
    cudaStream_t compute_stream = NULL, memory_stream = NULL;
    int n = mem_size / sizeof(float);
    double sequential_time = 0;
    cudaError_t err;

    err = cudaMalloc(&d_scratch, 1024 * sizeof(float));
    if (err != cudaSuccess) {
        sprintf(result->error_msg, "CUDA allocation failed: %s", cudaGetErrorString(err));
        return -1;
    }

    err = cudaMalloc(&d_src, mem_size);
    if (err != cudaSuccess) {
        cudaFree(d_scratch);
        sprintf(result->error_msg, "CUDA allocation failed: %s", cudaGetErrorString(err));
        return -1;
    }

    err = cudaMalloc(&d_dst, mem_size);
    if (err != cudaSuccess) {
        cudaFree(d_scratch);
        cudaFree(d_src);
        sprintf(result->error_msg, "CUDA allocation failed: %s", cudaGetErrorString(err));
        return -1;
    }

    /* Create streams for each workload type */
    cudaStreamCreate(&compute_stream);
    cudaStreamCreate(&memory_stream);

    /* Warmup */
    for (int i = 0; i < warmup; i++) {
        fcsp_kernel_compute_heavy<<<128, 256, 0, compute_stream>>>(d_scratch, 200);
        fcsp_kernel_memory_heavy<<<(n+255)/256, 256, 0, memory_stream>>>(d_src, d_dst, n);
    }
    cudaDeviceSynchronize();

    /* Measure sequential execution (compute then memory) */
    {
        timing_result_t t;
        timing_start(&t);

        fcsp_kernel_compute_heavy<<<128, 256>>>(d_scratch, 200);
        cudaDeviceSynchronize();
        fcsp_kernel_memory_heavy<<<(n+255)/256, 256>>>(d_src, d_dst, n);
        cudaDeviceSynchronize();

        timing_stop(&t);
        sequential_time = t.elapsed_ms;
    }

    /* Measure concurrent execution (compute + memory in parallel) */
    for (int i = 0; i < iterations; i++) {
        timing_result_t t;
        timing_start(&t);

        /* Launch complementary workloads concurrently */
        fcsp_kernel_compute_heavy<<<128, 256, 0, compute_stream>>>(d_scratch, 200);
        fcsp_kernel_memory_heavy<<<(n+255)/256, 256, 0, memory_stream>>>(d_src, d_dst, n);

        cudaStreamSynchronize(compute_stream);
        cudaStreamSynchronize(memory_stream);

        timing_stop(&t);

        /* Efficiency = how much faster concurrent is vs sequential */
        /* >100% means we benefit from running complementary workloads together */
        result->raw_values[i] = (sequential_time / t.elapsed_ms) * 100.0;
    }
    result->raw_count = iterations;

    /* Cleanup */
    cudaStreamDestroy(compute_stream);
    cudaStreamDestroy(memory_stream);
    cudaFree(d_dst);
    cudaFree(d_src);
    cudaFree(d_scratch);

    stats_calculate(result->raw_values, result->raw_count, &result->stats);

    result->value = result->stats.median;
    result->timestamp_ns = timing_get_ns();
    result->valid = true;
    result->success = true;

    printf("FCSP-001 Affinity Complementary: %.1f%% efficiency\n", result->value);
    printf("  (>100%% = benefit from running compute+memory together)\n");

    return 0;
}

/*
 * ============================================================================
 * FCSP-002: Workload Affinity - Conflicting Workload Detection
 * Tests if two compute-heavy workloads correctly cause contention
 * ============================================================================
 */

int bench_fcsp_affinity_conflicting(metric_result_t *result) {
    const int iterations = 50;
    const int warmup = 10;

    strcpy(result->metric_id, "FCSP-002");
    result->device_id = 0;
    result->valid = false;

    result->raw_values = (double*)malloc(sizeof(double) * iterations);
    if (!result->raw_values) {
        strcpy(result->error_msg, "Failed to allocate result storage");
        return -1;
    }
    result->raw_count = 0;

    float *d_scratch1, *d_scratch2;
    cudaError_t err;

    err = cudaMalloc(&d_scratch1, 1024 * sizeof(float));
    if (err != cudaSuccess) {
        sprintf(result->error_msg, "cudaMalloc failed: %s", cudaGetErrorString(err));
        return -1;
    }

    err = cudaMalloc(&d_scratch2, 1024 * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_scratch1);
        sprintf(result->error_msg, "cudaMalloc failed: %s", cudaGetErrorString(err));
        return -1;
    }

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    /* Warmup */
    for (int i = 0; i < warmup; i++) {
        fcsp_kernel_compute_heavy<<<128, 256, 0, stream1>>>(d_scratch1, 200);
        fcsp_kernel_compute_heavy<<<128, 256, 0, stream2>>>(d_scratch2, 200);
    }
    cudaDeviceSynchronize();

    /* Measure single compute workload */
    double single_time = 0;
    {
        timing_result_t t;
        timing_start(&t);
        fcsp_kernel_compute_heavy<<<128, 256>>>(d_scratch1, 200);
        cudaDeviceSynchronize();
        timing_stop(&t);
        single_time = t.elapsed_ms;
    }

    /* Measure two conflicting compute workloads */
    for (int i = 0; i < iterations; i++) {
        timing_result_t t;
        timing_start(&t);

        /* Both are compute-heavy - should conflict */
        fcsp_kernel_compute_heavy<<<128, 256, 0, stream1>>>(d_scratch1, 200);
        fcsp_kernel_compute_heavy<<<128, 256, 0, stream2>>>(d_scratch2, 200);

        cudaStreamSynchronize(stream1);
        cudaStreamSynchronize(stream2);

        timing_stop(&t);

        /* For conflicting workloads, concurrent time should be ~2x single */
        /* Ratio near 1.0 = severe conflict, near 2.0 = good parallelism */
        result->raw_values[i] = t.elapsed_ms / single_time;
    }
    result->raw_count = iterations;

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFree(d_scratch1);
    cudaFree(d_scratch2);

    stats_calculate(result->raw_values, result->raw_count, &result->stats);

    /* Conflict ratio: closer to 2.0 means workloads serialized (conflict detected) */
    result->value = result->stats.median;
    result->timestamp_ns = timing_get_ns();
    result->valid = true;
    result->success = true;

    printf("FCSP-002 Affinity Conflicting: %.2fx slowdown\n", result->value);
    printf("  (~2.0x = correctly detected conflict, workloads serialized)\n");

    return 0;
}

/*
 * ============================================================================
 * FCSP-003: Adaptive SM - Utilization Response Time
 * Tests how quickly SM limits adapt to changing utilization
 * ============================================================================
 */

int bench_fcsp_adaptive_sm_response(metric_result_t *result) {
    const int iterations = 20;
    const int warmup = 5;

    strcpy(result->metric_id, "FCSP-003");
    result->device_id = 0;
    result->valid = false;

    result->raw_values = (double*)malloc(sizeof(double) * iterations);
    if (!result->raw_values) {
        strcpy(result->error_msg, "Failed to allocate result storage");
        return -1;
    }
    result->raw_count = 0;

    float *d_data;
    cudaError_t err = cudaMalloc(&d_data, 32 * 1024 * 1024);
    if (err != cudaSuccess) {
        sprintf(result->error_msg, "cudaMalloc failed: %s", cudaGetErrorString(err));
        return -1;
    }

    int n = 32 * 1024 * 1024 / sizeof(float);

    /* Warmup */
    for (int i = 0; i < warmup; i++) {
        fcsp_kernel_balanced<<<(n+255)/256, 256>>>(d_data, n, 10);
    }
    cudaDeviceSynchronize();

    for (int i = 0; i < iterations; i++) {
        /* Phase 1: Light load (should trigger SM limit relaxation) */
        timing_result_t t_light;
        timing_start(&t_light);
        fcsp_kernel_balanced<<<64, 256>>>(d_data, n/16, 5);  /* Small workload */
        cudaDeviceSynchronize();
        timing_stop(&t_light);

        /* Wait for adaptive system to potentially adjust (200ms poll interval) */
        usleep(250000);  /* 250ms */

        /* Phase 2: Heavy load (measure if we got more resources) */
        timing_result_t t_heavy;
        timing_start(&t_heavy);
        fcsp_kernel_balanced<<<(n+255)/256, 256>>>(d_data, n, 10);
        cudaDeviceSynchronize();
        timing_stop(&t_heavy);

        /* Store the heavy workload time - should be faster if adaptive SM worked */
        result->raw_values[i] = t_heavy.elapsed_ms;
    }
    result->raw_count = iterations;

    cudaFree(d_data);

    stats_calculate(result->raw_values, result->raw_count, &result->stats);

    result->value = result->stats.median;
    result->timestamp_ns = timing_get_ns();
    result->valid = true;
    result->success = true;

    printf("FCSP-003 Adaptive SM Response: %.2f ms (heavy workload after light)\n", result->value);
    printf("  (Lower = faster adaptation to available resources)\n");

    return 0;
}

/*
 * ============================================================================
 * FCSP-004: Adaptive SM - Utilization Efficiency
 * Tests if adaptive SM improves throughput vs static limits
 * ============================================================================
 */

int bench_fcsp_adaptive_sm_efficiency(metric_result_t *result) {
    const int iterations = 30;
    const int warmup = 10;

    strcpy(result->metric_id, "FCSP-004");
    result->device_id = 0;
    result->valid = false;

    result->raw_values = (double*)malloc(sizeof(double) * iterations);
    if (!result->raw_values) {
        strcpy(result->error_msg, "Failed to allocate result storage");
        return -1;
    }
    result->raw_count = 0;

    float *d_data;
    cudaError_t err = cudaMalloc(&d_data, 64 * 1024 * 1024);
    if (err != cudaSuccess) {
        sprintf(result->error_msg, "cudaMalloc failed: %s", cudaGetErrorString(err));
        return -1;
    }

    int n = 64 * 1024 * 1024 / sizeof(float);

    /* Create multiple streams to simulate bursty workload */
    const int NUM_STREAMS = 4;
    cudaStream_t streams[NUM_STREAMS];
    for (int s = 0; s < NUM_STREAMS; s++) {
        cudaStreamCreate(&streams[s]);
    }

    /* Warmup */
    for (int i = 0; i < warmup; i++) {
        for (int s = 0; s < NUM_STREAMS; s++) {
            fcsp_kernel_balanced<<<(n/NUM_STREAMS+255)/256, 256, 0, streams[s]>>>(
                d_data + s * (n/NUM_STREAMS), n/NUM_STREAMS, 10);
        }
    }
    cudaDeviceSynchronize();

    /* Measure bursty workload pattern */
    for (int i = 0; i < iterations; i++) {
        timing_result_t t;
        timing_start(&t);

        /* Burst: all streams active */
        for (int s = 0; s < NUM_STREAMS; s++) {
            fcsp_kernel_balanced<<<(n/NUM_STREAMS+255)/256, 256, 0, streams[s]>>>(
                d_data + s * (n/NUM_STREAMS), n/NUM_STREAMS, 10);
        }

        for (int s = 0; s < NUM_STREAMS; s++) {
            cudaStreamSynchronize(streams[s]);
        }

        timing_stop(&t);

        /* Throughput in GB/s */
        double bytes = n * sizeof(float) * 2;  /* Read + write */
        result->raw_values[i] = (bytes / 1e9) / (t.elapsed_ms / 1000.0);
    }
    result->raw_count = iterations;

    for (int s = 0; s < NUM_STREAMS; s++) {
        cudaStreamDestroy(streams[s]);
    }
    cudaFree(d_data);

    stats_calculate(result->raw_values, result->raw_count, &result->stats);

    result->value = result->stats.median;
    result->timestamp_ns = timing_get_ns();
    result->valid = true;
    result->success = true;

    printf("FCSP-004 Adaptive SM Efficiency: %.2f GB/s throughput\n", result->value);
    printf("  (Higher = better utilization of available SM resources)\n");

    return 0;
}

/*
 * ============================================================================
 * FCSP-005: UVM Allocation Overhead
 * Measures overhead of UVM allocations vs native allocations
 * ============================================================================
 */

int bench_fcsp_uvm_alloc_overhead(metric_result_t *result) {
    const int iterations = 100;
    const int warmup = 20;
    const size_t alloc_size = 16 * 1024 * 1024;  /* 16MB */

    strcpy(result->metric_id, "FCSP-005");
    result->device_id = 0;
    result->valid = false;

    result->raw_values = (double*)malloc(sizeof(double) * iterations);
    if (!result->raw_values) {
        strcpy(result->error_msg, "Failed to allocate result storage");
        return -1;
    }
    result->raw_count = 0;

    /* Warmup native allocations */
    for (int i = 0; i < warmup; i++) {
        void *ptr;
        cudaMalloc(&ptr, alloc_size);
        cudaFree(ptr);
    }

    /* Measure native allocation time */
    double native_time = 0;
    {
        timing_result_t t;
        void *ptr;
        timing_start(&t);
        cudaMalloc(&ptr, alloc_size);
        timing_stop(&t);
        cudaFree(ptr);
        native_time = t.elapsed_us;
    }

    /* Warmup managed allocations */
    for (int i = 0; i < warmup; i++) {
        void *ptr;
        cudaMallocManaged(&ptr, alloc_size, cudaMemAttachGlobal);
        cudaFree(ptr);
    }

    /* Measure UVM allocation overhead */
    for (int i = 0; i < iterations; i++) {
        timing_result_t t;
        void *ptr;

        timing_start(&t);
        cudaError_t err = cudaMallocManaged(&ptr, alloc_size, cudaMemAttachGlobal);
        timing_stop(&t);

        if (err == cudaSuccess) {
            cudaFree(ptr);
            /* Store overhead ratio: UVM time / native time */
            result->raw_values[result->raw_count++] = t.elapsed_us / native_time;
        }
    }

    if (result->raw_count == 0) {
        strcpy(result->error_msg, "No successful UVM allocations");
        return -1;
    }

    stats_calculate(result->raw_values, result->raw_count, &result->stats);

    result->value = result->stats.median;
    result->timestamp_ns = timing_get_ns();
    result->valid = true;
    result->success = true;

    printf("FCSP-005 UVM Alloc Overhead: %.2fx vs native\n", result->value);
    printf("  (1.0x = same as native, >1.0x = slower)\n");

    return 0;
}

/*
 * ============================================================================
 * FCSP-006: UVM Prefetch Effectiveness
 * Tests if prefetch eliminates page fault latency
 * ============================================================================
 */

int bench_fcsp_uvm_prefetch_effectiveness(metric_result_t *result) {
    const int iterations = 50;
    const int warmup = 10;
    const size_t data_size = 64 * 1024 * 1024;  /* 64MB */

    strcpy(result->metric_id, "FCSP-006");
    result->device_id = 0;
    result->valid = false;

    result->raw_values = (double*)malloc(sizeof(double) * iterations);
    if (!result->raw_values) {
        strcpy(result->error_msg, "Failed to allocate result storage");
        return -1;
    }
    result->raw_count = 0;

    int device;
    cudaGetDevice(&device);

    float *d_data;
    cudaError_t err = cudaMallocManaged(&d_data, data_size, cudaMemAttachGlobal);
    if (err != cudaSuccess) {
        sprintf(result->error_msg, "cudaMallocManaged failed: %s", cudaGetErrorString(err));
        return -1;
    }

    int n = data_size / sizeof(float);

    /* Initialize data on CPU */
    for (int i = 0; i < n; i++) {
        d_data[i] = (float)i;
    }

    /* Warmup */
    for (int i = 0; i < warmup; i++) {
        cudaMemPrefetchAsync(d_data, data_size, device, 0);
        fcsp_kernel_balanced<<<(n+255)/256, 256>>>(d_data, n, 5);
        cudaDeviceSynchronize();

        /* Move back to CPU */
        cudaMemPrefetchAsync(d_data, data_size, cudaCpuDeviceId, 0);
        cudaDeviceSynchronize();
    }

    for (int i = 0; i < iterations; i++) {
        /* Ensure data is on CPU first */
        cudaMemPrefetchAsync(d_data, data_size, cudaCpuDeviceId, 0);
        cudaDeviceSynchronize();

        /* Measure WITHOUT prefetch (page faults expected) */
        timing_result_t t_no_prefetch;
        timing_start(&t_no_prefetch);
        fcsp_kernel_balanced<<<(n+255)/256, 256>>>(d_data, n, 5);
        cudaDeviceSynchronize();
        timing_stop(&t_no_prefetch);

        /* Move back to CPU */
        cudaMemPrefetchAsync(d_data, data_size, cudaCpuDeviceId, 0);
        cudaDeviceSynchronize();

        /* Measure WITH prefetch (no page faults) */
        timing_result_t t_with_prefetch;
        cudaMemPrefetchAsync(d_data, data_size, device, 0);
        cudaDeviceSynchronize();  /* Wait for prefetch to complete */

        timing_start(&t_with_prefetch);
        fcsp_kernel_balanced<<<(n+255)/256, 256>>>(d_data, n, 5);
        cudaDeviceSynchronize();
        timing_stop(&t_with_prefetch);

        /* Store speedup from prefetch */
        result->raw_values[result->raw_count++] = t_no_prefetch.elapsed_ms / t_with_prefetch.elapsed_ms;
    }

    cudaFree(d_data);

    stats_calculate(result->raw_values, result->raw_count, &result->stats);

    result->value = result->stats.median;
    result->timestamp_ns = timing_get_ns();
    result->valid = true;
    result->success = true;

    printf("FCSP-006 UVM Prefetch Effectiveness: %.2fx speedup\n", result->value);
    printf("  (>1.0x = prefetch eliminates page fault latency)\n");

    return 0;
}

/*
 * ============================================================================
 * FCSP-007: UVM CPU-GPU Transfer Overlap
 * Tests if data transfers can be hidden behind compute
 * ============================================================================
 */

int bench_fcsp_uvm_transfer_overlap(metric_result_t *result) {
    const int iterations = 30;
    const int warmup = 10;
    const size_t data_size = 32 * 1024 * 1024;  /* 32MB */

    strcpy(result->metric_id, "FCSP-007");
    result->device_id = 0;
    result->valid = false;

    result->raw_values = (double*)malloc(sizeof(double) * iterations);
    if (!result->raw_values) {
        strcpy(result->error_msg, "Failed to allocate result storage");
        return -1;
    }
    result->raw_count = 0;

    int device;
    cudaGetDevice(&device);

    /* Allocate two UVM buffers - one for compute, one for transfer */
    float *d_compute, *d_transfer;
    cudaError_t err = cudaMallocManaged(&d_compute, data_size, cudaMemAttachGlobal);
    if (err != cudaSuccess) {
        sprintf(result->error_msg, "cudaMallocManaged compute failed");
        return -1;
    }

    err = cudaMallocManaged(&d_transfer, data_size, cudaMemAttachGlobal);
    if (err != cudaSuccess) {
        cudaFree(d_compute);
        sprintf(result->error_msg, "cudaMallocManaged transfer failed");
        return -1;
    }

    int n = data_size / sizeof(float);

    /* Initialize */
    cudaMemPrefetchAsync(d_compute, data_size, device, 0);
    cudaDeviceSynchronize();

    cudaStream_t compute_stream, transfer_stream;
    cudaStreamCreate(&compute_stream);
    cudaStreamCreate(&transfer_stream);

    /* Warmup */
    for (int i = 0; i < warmup; i++) {
        fcsp_kernel_compute_heavy<<<128, 256, 0, compute_stream>>>(d_compute, 300);
        cudaMemPrefetchAsync(d_transfer, data_size, device, transfer_stream);
    }
    cudaDeviceSynchronize();

    for (int i = 0; i < iterations; i++) {
        /* Move transfer buffer to CPU */
        cudaMemPrefetchAsync(d_transfer, data_size, cudaCpuDeviceId, 0);
        cudaDeviceSynchronize();

        /* Measure sequential: compute then transfer */
        timing_result_t t_seq;
        timing_start(&t_seq);
        fcsp_kernel_compute_heavy<<<128, 256>>>(d_compute, 300);
        cudaDeviceSynchronize();
        cudaMemPrefetchAsync(d_transfer, data_size, device, 0);
        cudaDeviceSynchronize();
        timing_stop(&t_seq);

        /* Move back to CPU for next test */
        cudaMemPrefetchAsync(d_transfer, data_size, cudaCpuDeviceId, 0);
        cudaDeviceSynchronize();

        /* Measure overlapped: compute and transfer concurrently */
        timing_result_t t_overlap;
        timing_start(&t_overlap);
        fcsp_kernel_compute_heavy<<<128, 256, 0, compute_stream>>>(d_compute, 300);
        cudaMemPrefetchAsync(d_transfer, data_size, device, transfer_stream);
        cudaStreamSynchronize(compute_stream);
        cudaStreamSynchronize(transfer_stream);
        timing_stop(&t_overlap);

        /* Store overlap efficiency */
        result->raw_values[result->raw_count++] = (t_seq.elapsed_ms / t_overlap.elapsed_ms) * 100.0;
    }

    cudaStreamDestroy(compute_stream);
    cudaStreamDestroy(transfer_stream);
    cudaFree(d_compute);
    cudaFree(d_transfer);

    stats_calculate(result->raw_values, result->raw_count, &result->stats);

    result->value = result->stats.median;
    result->timestamp_ns = timing_get_ns();
    result->valid = true;
    result->success = true;

    printf("FCSP-007 UVM Transfer Overlap: %.1f%% efficiency\n", result->value);
    printf("  (>100%% = transfers hidden behind compute)\n");

    return 0;
}

/*
 * ============================================================================
 * FCSP-008: UVM Memory Pressure Handling
 * Tests behavior when GPU memory is oversubscribed
 * ============================================================================
 */

int bench_fcsp_uvm_memory_pressure(metric_result_t *result) {
    const int iterations = 20;

    strcpy(result->metric_id, "FCSP-008");
    result->device_id = 0;
    result->valid = false;

    result->raw_values = (double*)malloc(sizeof(double) * iterations);
    if (!result->raw_values) {
        strcpy(result->error_msg, "Failed to allocate result storage");
        return -1;
    }
    result->raw_count = 0;

    /* Get available GPU memory */
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    /* Try to allocate 150% of free memory using UVM */
    size_t alloc_size = (size_t)(free_mem * 0.3);  /* 30% chunks */
    int num_allocs = 5;  /* = 150% total */

    float **buffers = (float**)malloc(num_allocs * sizeof(float*));
    int successful_allocs = 0;

    int device;
    cudaGetDevice(&device);

    /* Allocate UVM memory exceeding GPU capacity */
    for (int i = 0; i < num_allocs; i++) {
        cudaError_t err = cudaMallocManaged(&buffers[i], alloc_size, cudaMemAttachGlobal);
        if (err == cudaSuccess) {
            successful_allocs++;
        } else {
            buffers[i] = NULL;
        }
    }

    if (successful_allocs < 3) {
        strcpy(result->error_msg, "Could not allocate enough UVM memory");
        for (int i = 0; i < num_allocs; i++) {
            if (buffers[i]) cudaFree(buffers[i]);
        }
        free(buffers);
        return -1;
    }

    int n = alloc_size / sizeof(float);

    /* Measure kernel execution under memory pressure */
    for (int iter = 0; iter < iterations; iter++) {
        /* Access each buffer to create memory pressure */
        timing_result_t t;
        timing_start(&t);

        for (int i = 0; i < successful_allocs; i++) {
            if (buffers[i]) {
                /* Prefetch to GPU (may cause eviction) */
                cudaMemPrefetchAsync(buffers[i], alloc_size, device, 0);
            }
        }
        cudaDeviceSynchronize();

        /* Run kernel on last buffer */
        if (buffers[successful_allocs-1]) {
            fcsp_kernel_balanced<<<(n+255)/256, 256>>>(buffers[successful_allocs-1], n, 5);
            cudaDeviceSynchronize();
        }

        timing_stop(&t);
        result->raw_values[result->raw_count++] = t.elapsed_ms;
    }

    /* Cleanup */
    for (int i = 0; i < num_allocs; i++) {
        if (buffers[i]) cudaFree(buffers[i]);
    }
    free(buffers);

    stats_calculate(result->raw_values, result->raw_count, &result->stats);

    /* Report: successful allocations beyond GPU capacity and execution time */
    result->value = result->stats.median;
    result->timestamp_ns = timing_get_ns();
    result->valid = true;
    result->success = true;

    printf("FCSP-008 UVM Memory Pressure: %.2f ms execution time\n", result->value);
    printf("  (Allocated %d x %.0fMB = %.0fMB with %.0fMB GPU free)\n",
           successful_allocs, alloc_size/1e6,
           successful_allocs * alloc_size/1e6, free_mem/1e6);

    return 0;
}

/*
 * ============================================================================
 * FCSP-009: Stream Activation Prefetch Latency
 * Tests latency of reactivating an idle stream with prefetch
 * ============================================================================
 */

int bench_fcsp_stream_activation_latency(metric_result_t *result) {
    const int iterations = 50;
    const int warmup = 10;
    const size_t data_size = 16 * 1024 * 1024;  /* 16MB per stream */
    const int idle_time_ms = 100;  /* Simulate 100ms idle period */

    strcpy(result->metric_id, "FCSP-009");
    result->device_id = 0;
    result->valid = false;

    result->raw_values = (double*)malloc(sizeof(double) * iterations);
    if (!result->raw_values) {
        strcpy(result->error_msg, "Failed to allocate result storage");
        return -1;
    }
    result->raw_count = 0;

    int device;
    cudaGetDevice(&device);

    float *d_data;
    cudaError_t err = cudaMallocManaged(&d_data, data_size, cudaMemAttachGlobal);
    if (err != cudaSuccess) {
        sprintf(result->error_msg, "cudaMallocManaged failed");
        return -1;
    }

    int n = data_size / sizeof(float);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    /* Initialize data on GPU */
    cudaMemPrefetchAsync(d_data, data_size, device, stream);
    cudaStreamSynchronize(stream);

    /* Warmup */
    for (int i = 0; i < warmup; i++) {
        fcsp_kernel_balanced<<<(n+255)/256, 256, 0, stream>>>(d_data, n, 5);
        cudaStreamSynchronize(stream);
    }

    for (int i = 0; i < iterations; i++) {
        /* Move data to CPU (simulate inactive stream) */
        cudaMemPrefetchAsync(d_data, data_size, cudaCpuDeviceId, stream);
        cudaStreamSynchronize(stream);

        /* Simulate idle period */
        usleep(idle_time_ms * 1000);

        /* Measure reactivation latency (includes prefetch + kernel) */
        timing_result_t t;
        timing_start(&t);

        /* This is what FCSP would do on stream reactivation */
        cudaMemPrefetchAsync(d_data, data_size, device, stream);
        fcsp_kernel_balanced<<<(n+255)/256, 256, 0, stream>>>(d_data, n, 5);
        cudaStreamSynchronize(stream);

        timing_stop(&t);
        result->raw_values[result->raw_count++] = t.elapsed_ms;
    }

    cudaStreamDestroy(stream);
    cudaFree(d_data);

    stats_calculate(result->raw_values, result->raw_count, &result->stats);

    result->value = result->stats.median;
    result->timestamp_ns = timing_get_ns();
    result->valid = true;
    result->success = true;

    printf("FCSP-009 Stream Activation Latency: %.2f ms\n", result->value);
    printf("  (Time to reactivate stream after %dms idle, including prefetch)\n", idle_time_ms);

    return 0;
}

/*
 * ============================================================================
 * FCSP-010: LLM Attention Pattern Efficiency
 * Tests efficiency for transformer attention workload patterns
 * ============================================================================
 */

int bench_fcsp_attention_pattern(metric_result_t *result) {
    const int iterations = 100;
    const int warmup = 20;
    const int num_heads = 32;
    const int head_dim = 128;
    const int seq_len = 512;

    strcpy(result->metric_id, "FCSP-010");
    result->device_id = 0;
    result->valid = false;

    result->raw_values = (double*)malloc(sizeof(double) * iterations);
    if (!result->raw_values) {
        strcpy(result->error_msg, "Failed to allocate result storage");
        return -1;
    }
    result->raw_count = 0;

    size_t qkv_size = num_heads * head_dim * sizeof(float);
    size_t out_size = num_heads * sizeof(float);

    float *d_qkv, *d_out;
    cudaError_t err = cudaMalloc(&d_qkv, qkv_size);
    if (err != cudaSuccess) {
        sprintf(result->error_msg, "cudaMalloc qkv failed");
        return -1;
    }

    err = cudaMalloc(&d_out, out_size);
    if (err != cudaSuccess) {
        cudaFree(d_qkv);
        sprintf(result->error_msg, "cudaMalloc out failed");
        return -1;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    /* Warmup */
    for (int i = 0; i < warmup; i++) {
        fcsp_kernel_attention_pattern<<<num_heads, head_dim, head_dim * sizeof(float), stream>>>(
            d_qkv, d_out, seq_len, head_dim);
    }
    cudaStreamSynchronize(stream);

    /* Measure attention pattern */
    for (int i = 0; i < iterations; i++) {
        timing_result_t t;
        timing_start(&t);

        fcsp_kernel_attention_pattern<<<num_heads, head_dim, head_dim * sizeof(float), stream>>>(
            d_qkv, d_out, seq_len, head_dim);
        cudaStreamSynchronize(stream);

        timing_stop(&t);

        /* Calculate TFLOPS */
        /* Attention FLOPs: 2 * num_heads * seq_len * head_dim (simplified) */
        double flops = 2.0 * num_heads * seq_len * head_dim;
        double tflops = (flops / 1e12) / (t.elapsed_ms / 1000.0);
        result->raw_values[result->raw_count++] = tflops;
    }

    cudaStreamDestroy(stream);
    cudaFree(d_qkv);
    cudaFree(d_out);

    stats_calculate(result->raw_values, result->raw_count, &result->stats);

    result->value = result->stats.median;
    result->timestamp_ns = timing_get_ns();
    result->valid = true;
    result->success = true;

    printf("FCSP-010 Attention Pattern: %.2f TFLOPS\n", result->value);
    printf("  (Heads=%d, HeadDim=%d, SeqLen=%d)\n", num_heads, head_dim, seq_len);

    return 0;
}

/*
 * ============================================================================
 * FCSP-011: LLM FFN Pattern Efficiency
 * Tests efficiency for transformer FFN workload patterns
 * ============================================================================
 */

int bench_fcsp_ffn_pattern(metric_result_t *result) {
    const int iterations = 100;
    const int warmup = 20;
    const int hidden_size = 4096;
    const int ffn_size = 11008;  /* LLaMA style 2.7x expansion */

    strcpy(result->metric_id, "FCSP-011");
    result->device_id = 0;
    result->valid = false;

    result->raw_values = (double*)malloc(sizeof(double) * iterations);
    if (!result->raw_values) {
        strcpy(result->error_msg, "Failed to allocate result storage");
        return -1;
    }
    result->raw_count = 0;

    size_t input_size = hidden_size * hidden_size * sizeof(float);
    size_t weight_size = hidden_size * ffn_size * sizeof(float);
    size_t output_size = hidden_size * ffn_size * sizeof(float);

    /* Declare all variables upfront */
    float *d_input = NULL, *d_weight = NULL, *d_output = NULL;
    cudaStream_t stream = NULL;
    dim3 block(16, 16);
    dim3 grid((ffn_size + 15) / 16, (hidden_size + 15) / 16);
    cudaError_t err;

    err = cudaMalloc(&d_input, input_size);
    if (err != cudaSuccess) {
        sprintf(result->error_msg, "CUDA allocation failed");
        return -1;
    }

    err = cudaMalloc(&d_weight, weight_size);
    if (err != cudaSuccess) {
        cudaFree(d_input);
        sprintf(result->error_msg, "CUDA allocation failed");
        return -1;
    }

    err = cudaMalloc(&d_output, output_size);
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_weight);
        sprintf(result->error_msg, "CUDA allocation failed");
        return -1;
    }

    cudaStreamCreate(&stream);

    /* Warmup */
    for (int i = 0; i < warmup; i++) {
        fcsp_kernel_ffn_pattern<<<grid, block, 0, stream>>>(
            d_input, d_weight, d_output, hidden_size, ffn_size);
    }
    cudaStreamSynchronize(stream);

    /* Measure FFN pattern */
    for (int i = 0; i < iterations; i++) {
        timing_result_t t;
        timing_start(&t);

        fcsp_kernel_ffn_pattern<<<grid, block, 0, stream>>>(
            d_input, d_weight, d_output, hidden_size, ffn_size);
        cudaStreamSynchronize(stream);

        timing_stop(&t);

        /* Calculate TFLOPS */
        /* FFN FLOPs: 2 * M * N * K (GEMM) + N (GELU) */
        double flops = 2.0 * hidden_size * ffn_size * hidden_size + ffn_size;
        double tflops = (flops / 1e12) / (t.elapsed_ms / 1000.0);
        result->raw_values[result->raw_count++] = tflops;
    }

    cudaStreamDestroy(stream);
    cudaFree(d_output);
    cudaFree(d_weight);
    cudaFree(d_input);

    stats_calculate(result->raw_values, result->raw_count, &result->stats);

    result->value = result->stats.median;
    result->timestamp_ns = timing_get_ns();
    result->valid = true;
    result->success = true;

    printf("FCSP-011 FFN Pattern: %.2f TFLOPS\n", result->value);
    printf("  (Hidden=%d, FFN=%d)\n", hidden_size, ffn_size);

    return 0;
}

/*
 * ============================================================================
 * FCSP-012: Mixed Workload Orchestration
 * Tests overall system efficiency with mixed LLM-style workloads
 * ============================================================================
 */

int bench_fcsp_mixed_workload(metric_result_t *result) {
    const int iterations = 30;
    const int warmup = 10;

    strcpy(result->metric_id, "FCSP-012");
    result->device_id = 0;
    result->valid = false;

    result->raw_values = (double*)malloc(sizeof(double) * iterations);
    if (!result->raw_values) {
        strcpy(result->error_msg, "Failed to allocate result storage");
        return -1;
    }
    result->raw_count = 0;

    /* Allocate buffers for different workload types */
    const size_t compute_size = 1024 * sizeof(float);
    const size_t memory_size = 32 * 1024 * 1024;
    const size_t balanced_size = 8 * 1024 * 1024;

    float *d_compute = NULL, *d_mem_src = NULL, *d_mem_dst = NULL, *d_balanced = NULL;
    cudaStream_t compute_stream = NULL, memory_stream = NULL, balanced_stream = NULL;
    int n_mem = memory_size / sizeof(float);
    int n_bal = balanced_size / sizeof(float);
    cudaError_t err;

    err = cudaMalloc(&d_compute, compute_size);
    if (err != cudaSuccess) {
        sprintf(result->error_msg, "CUDA allocation failed");
        return -1;
    }

    err = cudaMalloc(&d_mem_src, memory_size);
    if (err != cudaSuccess) {
        cudaFree(d_compute);
        sprintf(result->error_msg, "CUDA allocation failed");
        return -1;
    }

    err = cudaMalloc(&d_mem_dst, memory_size);
    if (err != cudaSuccess) {
        cudaFree(d_compute);
        cudaFree(d_mem_src);
        sprintf(result->error_msg, "CUDA allocation failed");
        return -1;
    }

    err = cudaMalloc(&d_balanced, balanced_size);
    if (err != cudaSuccess) {
        cudaFree(d_compute);
        cudaFree(d_mem_src);
        cudaFree(d_mem_dst);
        sprintf(result->error_msg, "CUDA allocation failed");
        return -1;
    }

    /* Create streams for each workload type */
    cudaStreamCreate(&compute_stream);
    cudaStreamCreate(&memory_stream);
    cudaStreamCreate(&balanced_stream);

    /* Warmup all workload types */
    for (int i = 0; i < warmup; i++) {
        fcsp_kernel_compute_heavy<<<128, 256, 0, compute_stream>>>(d_compute, 100);
        fcsp_kernel_memory_heavy<<<(n_mem+255)/256, 256, 0, memory_stream>>>(d_mem_src, d_mem_dst, n_mem);
        fcsp_kernel_balanced<<<(n_bal+255)/256, 256, 0, balanced_stream>>>(d_balanced, n_bal, 10);
    }
    cudaDeviceSynchronize();

    /* Measure mixed workload execution */
    for (int i = 0; i < iterations; i++) {
        timing_result_t t;
        timing_start(&t);

        /* Launch all three workload types concurrently */
        fcsp_kernel_compute_heavy<<<128, 256, 0, compute_stream>>>(d_compute, 100);
        fcsp_kernel_memory_heavy<<<(n_mem+255)/256, 256, 0, memory_stream>>>(d_mem_src, d_mem_dst, n_mem);
        fcsp_kernel_balanced<<<(n_bal+255)/256, 256, 0, balanced_stream>>>(d_balanced, n_bal, 10);

        cudaStreamSynchronize(compute_stream);
        cudaStreamSynchronize(memory_stream);
        cudaStreamSynchronize(balanced_stream);

        timing_stop(&t);

        /* Total throughput (ops per second) */
        result->raw_values[result->raw_count++] = 3.0 / (t.elapsed_ms / 1000.0);
    }

    cudaStreamDestroy(compute_stream);
    cudaStreamDestroy(memory_stream);
    cudaStreamDestroy(balanced_stream);

    cudaFree(d_balanced);
    cudaFree(d_mem_dst);
    cudaFree(d_mem_src);
    cudaFree(d_compute);

    stats_calculate(result->raw_values, result->raw_count, &result->stats);

    result->value = result->stats.median;
    result->timestamp_ns = timing_get_ns();
    result->valid = true;
    result->success = true;

    printf("FCSP-012 Mixed Workload: %.1f ops/sec\n", result->value);
    printf("  (Concurrent compute + memory + balanced workloads)\n");

    return 0;
}

/*
 * ============================================================================
 * Run All FCSP Feature Metrics
 * ============================================================================
 */

void bench_run_fcsp_features(bench_config_t *config, metric_result_t *results, int *count) {
    (void)config;

    printf("\nRunning FCSP Feature benchmarks...\n");

    int idx = 0;

    /* Workload Affinity Metrics */
    printf("Running FCSP-001: Affinity Complementary Workloads\n");
    bench_fcsp_affinity_complementary(&results[idx++]);

    printf("Running FCSP-002: Affinity Conflicting Workloads\n");
    bench_fcsp_affinity_conflicting(&results[idx++]);

    /* Adaptive SM Metrics */
    printf("Running FCSP-003: Adaptive SM Response Time\n");
    bench_fcsp_adaptive_sm_response(&results[idx++]);

    printf("Running FCSP-004: Adaptive SM Efficiency\n");
    bench_fcsp_adaptive_sm_efficiency(&results[idx++]);

    /* UVM Metrics */
    printf("Running FCSP-005: UVM Allocation Overhead\n");
    bench_fcsp_uvm_alloc_overhead(&results[idx++]);

    printf("Running FCSP-006: UVM Prefetch Effectiveness\n");
    bench_fcsp_uvm_prefetch_effectiveness(&results[idx++]);

    printf("Running FCSP-007: UVM Transfer Overlap\n");
    bench_fcsp_uvm_transfer_overlap(&results[idx++]);

    printf("Running FCSP-008: UVM Memory Pressure\n");
    bench_fcsp_uvm_memory_pressure(&results[idx++]);

    printf("Running FCSP-009: Stream Activation Latency\n");
    bench_fcsp_stream_activation_latency(&results[idx++]);

    /* LLM Pattern Metrics */
    printf("Running FCSP-010: Attention Pattern Efficiency\n");
    bench_fcsp_attention_pattern(&results[idx++]);

    printf("Running FCSP-011: FFN Pattern Efficiency\n");
    bench_fcsp_ffn_pattern(&results[idx++]);

    printf("Running FCSP-012: Mixed Workload Orchestration\n");
    bench_fcsp_mixed_workload(&results[idx++]);

    *count = idx;

    printf("FCSP Feature benchmarks completed: %d/12 metrics\n\n", idx);
}
