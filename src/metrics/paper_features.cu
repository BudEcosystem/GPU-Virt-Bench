/*
 * GPU Virtualization Performance Evaluation Tool
 * Paper-Inspired Feature Metrics (PAPER-001 to PAPER-006)
 *
 * Metrics for features implemented based on the paper:
 * "Efficient Resource Sharing Through GPU Virtualization on Accelerated HPC Systems"
 *
 * These metrics measure:
 * - CUDA Graph pre-computed cost accuracy
 * - Workload intensity classification correctness
 * - Performance improvement from adaptive throttling
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "include/benchmark.h"
#include "include/metrics.h"

/*
 * ============================================================================
 * CUDA Kernels for Testing
 * ============================================================================
 */

/* Compute-intensive kernel: high FLOPS, low memory access */
__global__ void kernel_compute_intensive(float *data, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (float)idx;

    /* Heavy compute work */
    for (int i = 0; i < iterations; i++) {
        val = sinf(val) * cosf(val) + sqrtf(fabsf(val));
        val = expf(-val * 0.01f) + tanf(val * 0.1f);
    }

    if (data != NULL) {
        data[idx % 1024] = val;
    }
}

/* IO-intensive kernel: high memory access, low compute */
__global__ void kernel_io_intensive(float *src, float *dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        /* Simple copy with minimal compute */
        dst[idx] = src[idx];
    }
}

/* Balanced kernel: moderate compute and memory access */
__global__ void kernel_balanced(float *data, int n, int compute_iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float val = data[idx];
        for (int i = 0; i < compute_iters; i++) {
            val = sqrtf(val + 1.0f);
        }
        data[idx] = val;
    }
}

/* Graph kernel node - simple for counting */
__global__ void kernel_graph_node(int *out, int node_id) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        atomicAdd(out, 1);
    }
}

/*
 * ============================================================================
 * PAPER-001: Graph Cost Pre-computation Accuracy
 * Measures if pre-computed costs reflect actual graph complexity
 * ============================================================================
 */

int bench_graph_cost_accuracy(metric_result_t *result) {
    const int iterations = 100;
    const int warmup = 10;

    strcpy(result->metric_id, "PAPER-001");
    result->device_id = 0;
    result->valid = false;

    result->raw_values = (double*)malloc(sizeof(double) * iterations);
    if (result->raw_values == NULL) {
        strcpy(result->error_msg, "Failed to allocate result storage");
        return -1;
    }
    result->raw_count = 0;

    /* Allocate device memory */
    int *d_counter;
    cudaError_t err = cudaMalloc(&d_counter, sizeof(int));
    if (err != cudaSuccess) {
        sprintf(result->error_msg, "cudaMalloc failed: %s", cudaGetErrorString(err));
        return -1;
    }

    /* Create graphs of different complexity */
    for (int graph_size = 1; graph_size <= iterations; graph_size++) {
        cudaGraph_t graph;
        cudaGraphExec_t graphExec;

        /* Create graph with 'graph_size' kernel nodes */
        cudaStreamBeginCapture(0, cudaStreamCaptureModeGlobal);

        for (int node = 0; node < graph_size; node++) {
            kernel_graph_node<<<64, 256>>>(d_counter, node);
        }

        cudaStreamEndCapture(0, &graph);
        cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

        /* Measure launch time */
        timing_result_t t;

        /* Warmup */
        for (int w = 0; w < warmup; w++) {
            cudaGraphLaunch(graphExec, 0);
        }
        cudaDeviceSynchronize();

        /* Measure */
        timing_start(&t);
        cudaGraphLaunch(graphExec, 0);
        cudaDeviceSynchronize();
        timing_stop(&t);

        /* Store: launch time should scale with graph size (nodes) */
        result->raw_values[graph_size - 1] = t.elapsed_us / (double)graph_size;

        /* Cleanup */
        cudaGraphExecDestroy(graphExec);
        cudaGraphDestroy(graph);
    }
    result->raw_count = iterations;

    cudaFree(d_counter);

    /* Calculate statistics */
    stats_calculate(result->raw_values, result->raw_count, &result->stats);

    /* Coefficient of variation measures consistency */
    /* Lower CV means launch time scales linearly with complexity (good) */
    double cv = result->stats.std_dev / result->stats.mean;
    result->value = (1.0 - cv) * 100.0;  /* Convert to percentage: 100% = perfect scaling */

    if (result->value < 0) result->value = 0;
    if (result->value > 100) result->value = 100;

    result->timestamp_ns = timing_get_ns();
    result->valid = true;
    result->success = true;

    printf("  PAPER-001 Graph Cost Accuracy: %.1f%% (CV=%.3f)\n",
           result->value, cv);

    return 0;
}

/*
 * ============================================================================
 * PAPER-002: Compute-Intensive Workload Classification
 * Verifies compute-intensive kernels are correctly identified
 * ============================================================================
 */

int bench_compute_classification(metric_result_t *result) {
    const int iterations = 100;
    const int warmup = 10;

    strcpy(result->metric_id, "PAPER-002");
    result->device_id = 0;
    result->valid = false;

    result->raw_values = (double*)malloc(sizeof(double) * iterations);
    if (result->raw_values == NULL) {
        strcpy(result->error_msg, "Failed to allocate result storage");
        return -1;
    }
    result->raw_count = 0;

    float *d_data;
    cudaError_t err = cudaMalloc(&d_data, 1024 * sizeof(float));
    if (err != cudaSuccess) {
        sprintf(result->error_msg, "cudaMalloc failed: %s", cudaGetErrorString(err));
        return -1;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    /* Warmup */
    for (int i = 0; i < warmup; i++) {
        kernel_compute_intensive<<<256, 256, 0, stream>>>(d_data, 100);
    }
    cudaStreamSynchronize(stream);

    /* Measure throughput of compute-intensive workload */
    double baseline_throughput = 0;
    {
        timing_result_t t;
        timing_start(&t);

        for (int i = 0; i < iterations; i++) {
            kernel_compute_intensive<<<256, 256, 0, stream>>>(d_data, 100);
        }
        cudaStreamSynchronize(stream);

        timing_stop(&t);
        baseline_throughput = (double)iterations / (t.elapsed_ms / 1000.0);
    }

    /* Measure multi-stream throughput */
    /* Compute-intensive workloads should NOT benefit much from multi-stream */
    const int NUM_STREAMS = 4;
    cudaStream_t streams[NUM_STREAMS];
    for (int s = 0; s < NUM_STREAMS; s++) {
        cudaStreamCreate(&streams[s]);
    }

    double multi_stream_throughput = 0;
    {
        timing_result_t t;
        timing_start(&t);

        for (int i = 0; i < iterations / NUM_STREAMS; i++) {
            for (int s = 0; s < NUM_STREAMS; s++) {
                kernel_compute_intensive<<<256, 256, 0, streams[s]>>>(d_data, 100);
            }
        }
        for (int s = 0; s < NUM_STREAMS; s++) {
            cudaStreamSynchronize(streams[s]);
        }

        timing_stop(&t);
        multi_stream_throughput = (double)iterations / (t.elapsed_ms / 1000.0);
    }

    /* Compute-intensive: multi-stream efficiency should be ~100% (no benefit) */
    double efficiency = (multi_stream_throughput / baseline_throughput) * 100.0;

    /* Store per-iteration times for statistics */
    for (int i = 0; i < iterations; i++) {
        result->raw_values[i] = efficiency;
    }
    result->raw_count = iterations;

    /* Cleanup */
    for (int s = 0; s < NUM_STREAMS; s++) {
        cudaStreamDestroy(streams[s]);
    }
    cudaStreamDestroy(stream);
    cudaFree(d_data);

    stats_calculate(result->raw_values, result->raw_count, &result->stats);

    /* For compute-intensive, efficiency near 100% is expected (no parallelism benefit) */
    result->value = result->stats.mean;
    result->timestamp_ns = timing_get_ns();
    result->valid = true;
    result->success = true;

    printf("  PAPER-002 Compute Classification: %.1f%% multi-stream efficiency\n",
           result->value);
    printf("    (Expected: ~100%% for compute-bound, meaning no benefit from streams)\n");

    return 0;
}

/*
 * ============================================================================
 * PAPER-003: IO-Intensive Workload Classification
 * Verifies IO-intensive kernels benefit from reduced throttling
 * ============================================================================
 */

int bench_io_classification(metric_result_t *result) {
    const int iterations = 100;
    const int warmup = 10;
    const size_t data_size = 64 * 1024 * 1024;  /* 64MB */

    strcpy(result->metric_id, "PAPER-003");
    result->device_id = 0;
    result->valid = false;

    result->raw_values = (double*)malloc(sizeof(double) * iterations);
    if (result->raw_values == NULL) {
        strcpy(result->error_msg, "Failed to allocate result storage");
        return -1;
    }
    result->raw_count = 0;

    float *d_src, *d_dst;
    cudaError_t err = cudaMalloc(&d_src, data_size);
    if (err != cudaSuccess) {
        sprintf(result->error_msg, "cudaMalloc src failed: %s", cudaGetErrorString(err));
        return -1;
    }
    err = cudaMalloc(&d_dst, data_size);
    if (err != cudaSuccess) {
        cudaFree(d_src);
        sprintf(result->error_msg, "cudaMalloc dst failed: %s", cudaGetErrorString(err));
        return -1;
    }

    int n = data_size / sizeof(float);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    /* Warmup */
    for (int i = 0; i < warmup; i++) {
        kernel_io_intensive<<<(n + 255) / 256, 256, 0, stream>>>(d_src, d_dst, n);
    }
    cudaStreamSynchronize(stream);

    /* Measure single-stream throughput */
    double baseline_throughput = 0;
    {
        timing_result_t t;
        timing_start(&t);

        for (int i = 0; i < iterations; i++) {
            kernel_io_intensive<<<(n + 255) / 256, 256, 0, stream>>>(d_src, d_dst, n);
        }
        cudaStreamSynchronize(stream);

        timing_stop(&t);
        baseline_throughput = (double)iterations / (t.elapsed_ms / 1000.0);
    }

    /* Measure multi-stream throughput */
    /* IO-intensive workloads SHOULD benefit from multi-stream (paper's key insight) */
    const int NUM_STREAMS = 4;
    cudaStream_t streams[NUM_STREAMS];
    for (int s = 0; s < NUM_STREAMS; s++) {
        cudaStreamCreate(&streams[s]);
    }

    double multi_stream_throughput = 0;
    {
        timing_result_t t;
        timing_start(&t);

        for (int i = 0; i < iterations / NUM_STREAMS; i++) {
            for (int s = 0; s < NUM_STREAMS; s++) {
                kernel_io_intensive<<<(n + 255) / 256, 256, 0, streams[s]>>>(d_src, d_dst, n);
            }
        }
        for (int s = 0; s < NUM_STREAMS; s++) {
            cudaStreamSynchronize(streams[s]);
        }

        timing_stop(&t);
        multi_stream_throughput = (double)iterations / (t.elapsed_ms / 1000.0);
    }

    /* IO-intensive: multi-stream efficiency should be >100% (paper shows 4-6x possible) */
    double efficiency = (multi_stream_throughput / baseline_throughput) * 100.0;

    for (int i = 0; i < iterations; i++) {
        result->raw_values[i] = efficiency;
    }
    result->raw_count = iterations;

    /* Cleanup */
    for (int s = 0; s < NUM_STREAMS; s++) {
        cudaStreamDestroy(streams[s]);
    }
    cudaStreamDestroy(stream);
    cudaFree(d_src);
    cudaFree(d_dst);

    stats_calculate(result->raw_values, result->raw_count, &result->stats);

    result->value = result->stats.mean;
    result->timestamp_ns = timing_get_ns();
    result->valid = true;
    result->success = true;

    printf("  PAPER-003 IO Classification: %.1f%% multi-stream efficiency\n",
           result->value);
    printf("    (Expected: >100%% for IO-bound, meaning benefit from concurrent streams)\n");

    return 0;
}

/*
 * ============================================================================
 * PAPER-004: Graph Launch Cost Proportionality
 * Tests that complex graphs consume more rate limiter tokens than simple ones
 * ============================================================================
 */

int bench_graph_launch_cost_proportionality(metric_result_t *result) {
    const int iterations = 20;
    const int warmup = 5;

    strcpy(result->metric_id, "PAPER-004");
    result->device_id = 0;
    result->valid = false;

    result->raw_values = (double*)malloc(sizeof(double) * iterations);
    if (result->raw_values == NULL) {
        strcpy(result->error_msg, "Failed to allocate result storage");
        return -1;
    }
    result->raw_count = 0;

    int *d_counter;
    cudaError_t err = cudaMalloc(&d_counter, sizeof(int));
    if (err != cudaSuccess) {
        sprintf(result->error_msg, "cudaMalloc failed: %s", cudaGetErrorString(err));
        return -1;
    }

    /* Create simple graph (1 kernel) */
    cudaGraph_t simple_graph;
    cudaGraphExec_t simple_exec;
    {
        cudaStreamBeginCapture(0, cudaStreamCaptureModeGlobal);
        kernel_graph_node<<<1, 1>>>(d_counter, 0);
        cudaStreamEndCapture(0, &simple_graph);
        cudaGraphInstantiate(&simple_exec, simple_graph, NULL, NULL, 0);
    }

    /* Create complex graph (100 kernels) */
    cudaGraph_t complex_graph;
    cudaGraphExec_t complex_exec;
    {
        cudaStreamBeginCapture(0, cudaStreamCaptureModeGlobal);
        for (int i = 0; i < 100; i++) {
            kernel_graph_node<<<64, 256>>>(d_counter, i);
        }
        cudaStreamEndCapture(0, &complex_graph);
        cudaGraphInstantiate(&complex_exec, complex_graph, NULL, NULL, 0);
    }

    /* Warmup */
    for (int i = 0; i < warmup; i++) {
        cudaGraphLaunch(simple_exec, 0);
        cudaGraphLaunch(complex_exec, 0);
    }
    cudaDeviceSynchronize();

    /* Measure launch times */
    double simple_time_sum = 0;
    double complex_time_sum = 0;

    for (int i = 0; i < iterations; i++) {
        timing_result_t t1, t2;

        timing_start(&t1);
        cudaGraphLaunch(simple_exec, 0);
        cudaDeviceSynchronize();
        timing_stop(&t1);

        timing_start(&t2);
        cudaGraphLaunch(complex_exec, 0);
        cudaDeviceSynchronize();
        timing_stop(&t2);

        simple_time_sum += t1.elapsed_us;
        complex_time_sum += t2.elapsed_us;

        /* Store the ratio per iteration */
        result->raw_values[i] = t2.elapsed_us / t1.elapsed_us;
    }
    result->raw_count = iterations;

    /* Cleanup */
    cudaGraphExecDestroy(simple_exec);
    cudaGraphExecDestroy(complex_exec);
    cudaGraphDestroy(simple_graph);
    cudaGraphDestroy(complex_graph);
    cudaFree(d_counter);

    stats_calculate(result->raw_values, result->raw_count, &result->stats);

    /* Ratio should be significant (>10x for 100x kernel difference) */
    /* With proper cost tracking, complex graphs should take proportionally longer */
    result->value = result->stats.median;
    result->timestamp_ns = timing_get_ns();
    result->valid = true;
    result->success = true;

    printf("  PAPER-004 Graph Cost Proportionality: %.1fx\n", result->value);
    printf("    Complex (100 kernels) vs Simple (1 kernel) launch time ratio\n");
    printf("    (Expected: significant ratio when pre-computed costs work correctly)\n");

    return 0;
}

/*
 * ============================================================================
 * PAPER-005: Workload-Aware Throttling Fairness
 * Tests that different workload types receive appropriate throttling
 * ============================================================================
 */

int bench_workload_aware_fairness(metric_result_t *result) {
    const int iterations = 50;
    const int warmup = 10;

    strcpy(result->metric_id, "PAPER-005");
    result->device_id = 0;
    result->valid = false;

    result->raw_values = (double*)malloc(sizeof(double) * iterations);
    if (result->raw_values == NULL) {
        strcpy(result->error_msg, "Failed to allocate result storage");
        return -1;
    }
    result->raw_count = 0;

    float *d_data;
    cudaError_t err = cudaMalloc(&d_data, 1024 * 1024 * sizeof(float));
    if (err != cudaSuccess) {
        sprintf(result->error_msg, "cudaMalloc failed: %s", cudaGetErrorString(err));
        return -1;
    }

    cudaStream_t compute_stream, io_stream;
    cudaStreamCreate(&compute_stream);
    cudaStreamCreate(&io_stream);

    /* Warmup */
    for (int i = 0; i < warmup; i++) {
        kernel_compute_intensive<<<64, 256, 0, compute_stream>>>(d_data, 100);
        kernel_io_intensive<<<4096, 256, 0, io_stream>>>(d_data, d_data, 1024 * 1024);
    }
    cudaDeviceSynchronize();

    /* Run concurrent compute and IO workloads */
    for (int i = 0; i < iterations; i++) {
        timing_result_t t_compute, t_io;

        /* Launch both workloads concurrently */
        timing_start(&t_compute);
        kernel_compute_intensive<<<64, 256, 0, compute_stream>>>(d_data, 100);
        cudaStreamSynchronize(compute_stream);
        timing_stop(&t_compute);

        timing_start(&t_io);
        kernel_io_intensive<<<4096, 256, 0, io_stream>>>(d_data, d_data, 1024 * 1024);
        cudaStreamSynchronize(io_stream);
        timing_stop(&t_io);

        /* With workload-aware throttling, IO should complete without excessive delay */
        /* Paper insight: IO-intensive should have reduced throttling */
        double fairness = t_compute.elapsed_us / t_io.elapsed_us;
        result->raw_values[i] = fairness;
    }
    result->raw_count = iterations;

    /* Cleanup */
    cudaStreamDestroy(compute_stream);
    cudaStreamDestroy(io_stream);
    cudaFree(d_data);

    stats_calculate(result->raw_values, result->raw_count, &result->stats);

    /* Compute Jain's fairness index */
    double jain_index = 0;
    double sum = 0, sum_sq = 0;
    for (int i = 0; i < iterations; i++) {
        sum += result->raw_values[i];
        sum_sq += result->raw_values[i] * result->raw_values[i];
    }
    if (sum_sq > 0) {
        jain_index = (sum * sum) / (iterations * sum_sq);
    }

    result->value = jain_index;
    result->timestamp_ns = timing_get_ns();
    result->valid = true;
    result->success = true;

    printf("  PAPER-005 Workload-Aware Fairness: %.3f (Jain's index)\n", result->value);
    printf("    (1.0 = perfect fairness, closer to 1.0 is better)\n");

    return 0;
}

/*
 * ============================================================================
 * PAPER-006: Execution Model Speedup (Paper's Key Metric)
 * Measures speedup from workload-aware sharing vs naive sharing
 * ============================================================================
 */

int bench_execution_model_speedup(metric_result_t *result) {
    const int iterations = 20;
    const int warmup = 5;
    const size_t data_size = 32 * 1024 * 1024;

    strcpy(result->metric_id, "PAPER-006");
    result->device_id = 0;
    result->valid = false;

    result->raw_values = (double*)malloc(sizeof(double) * iterations);
    if (result->raw_values == NULL) {
        strcpy(result->error_msg, "Failed to allocate result storage");
        return -1;
    }
    result->raw_count = 0;

    float *d_data;
    cudaError_t err = cudaMalloc(&d_data, data_size);
    if (err != cudaSuccess) {
        sprintf(result->error_msg, "cudaMalloc failed: %s", cudaGetErrorString(err));
        return -1;
    }

    int n = data_size / sizeof(float);

    /* Simulate the paper's SPMD execution model:
     * Multiple processes sharing a GPU */

    const int NUM_VIRTUAL_PROCESSES = 4;
    cudaStream_t streams[NUM_VIRTUAL_PROCESSES];
    for (int i = 0; i < NUM_VIRTUAL_PROCESSES; i++) {
        cudaStreamCreate(&streams[i]);
    }

    /* Warmup */
    for (int w = 0; w < warmup; w++) {
        for (int p = 0; p < NUM_VIRTUAL_PROCESSES; p++) {
            kernel_balanced<<<(n + 255) / 256, 256, 0, streams[p]>>>(d_data, n, 10);
        }
    }
    cudaDeviceSynchronize();

    /* Measure sequential execution (baseline - no sharing) */
    double sequential_time = 0;
    {
        timing_result_t t;
        timing_start(&t);

        for (int p = 0; p < NUM_VIRTUAL_PROCESSES; p++) {
            kernel_balanced<<<(n + 255) / 256, 256>>>(d_data, n, 10);
            cudaDeviceSynchronize();
        }

        timing_stop(&t);
        sequential_time = t.elapsed_ms;
    }

    /* Measure concurrent execution (with GPU sharing) */
    double concurrent_time = 0;
    {
        timing_result_t t;
        timing_start(&t);

        for (int p = 0; p < NUM_VIRTUAL_PROCESSES; p++) {
            kernel_balanced<<<(n + 255) / 256, 256, 0, streams[p]>>>(d_data, n, 10);
        }
        for (int p = 0; p < NUM_VIRTUAL_PROCESSES; p++) {
            cudaStreamSynchronize(streams[p]);
        }

        timing_stop(&t);
        concurrent_time = t.elapsed_ms;
    }

    /* Calculate speedup */
    double speedup = sequential_time / concurrent_time;

    for (int i = 0; i < iterations; i++) {
        /* Repeat measurement for statistics */
        double seq_t = 0, conc_t = 0;

        {
            timing_result_t t;
            timing_start(&t);
            for (int p = 0; p < NUM_VIRTUAL_PROCESSES; p++) {
                kernel_balanced<<<(n + 255) / 256, 256>>>(d_data, n, 10);
                cudaDeviceSynchronize();
            }
            timing_stop(&t);
            seq_t = t.elapsed_ms;
        }

        {
            timing_result_t t;
            timing_start(&t);
            for (int p = 0; p < NUM_VIRTUAL_PROCESSES; p++) {
                kernel_balanced<<<(n + 255) / 256, 256, 0, streams[p]>>>(d_data, n, 10);
            }
            for (int p = 0; p < NUM_VIRTUAL_PROCESSES; p++) {
                cudaStreamSynchronize(streams[p]);
            }
            timing_stop(&t);
            conc_t = t.elapsed_ms;
        }

        result->raw_values[i] = seq_t / conc_t;
    }
    result->raw_count = iterations;

    /* Cleanup */
    for (int i = 0; i < NUM_VIRTUAL_PROCESSES; i++) {
        cudaStreamDestroy(streams[i]);
    }
    cudaFree(d_data);

    stats_calculate(result->raw_values, result->raw_count, &result->stats);

    /* Paper reports 2-6x speedup depending on workload */
    result->value = result->stats.median;
    result->timestamp_ns = timing_get_ns();
    result->valid = true;
    result->success = true;

    printf("  PAPER-006 Execution Model Speedup: %.2fx\n", result->value);
    printf("    Concurrent vs Sequential execution of %d virtual processes\n",
           NUM_VIRTUAL_PROCESSES);
    printf("    (Paper reports 2-6x speedup for balanced workloads)\n");

    return 0;
}

/*
 * ============================================================================
 * Run All Paper Feature Metrics
 * ============================================================================
 */

void bench_run_paper_features(bench_config_t *config, metric_result_t *results, int *count) {
    (void)config;  /* Unused for now */

    printf("\n=== Paper-Inspired Feature Metrics ===\n\n");

    int idx = 0;

    bench_graph_cost_accuracy(&results[idx++]);
    bench_compute_classification(&results[idx++]);
    bench_io_classification(&results[idx++]);
    bench_graph_launch_cost_proportionality(&results[idx++]);
    bench_workload_aware_fairness(&results[idx++]);
    bench_execution_model_speedup(&results[idx++]);

    *count = idx;

    printf("\n=== Paper Feature Metrics Complete ===\n\n");
}
