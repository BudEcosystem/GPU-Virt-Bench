/**
 * Multi-Graph Multi-Tenant Manager Benchmarks
 * Performance evaluation of SLO-aware graph scheduling
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#include "../../../bud_fcsp/src/graph/multi_graph_manager.h"
#include "../../../bud_fcsp/src/graph/multi_graph_types.h"

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

static inline uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* ============================================================================
 * Test Kernels
 * ============================================================================ */

__global__ void compute_kernel(float *data, int size, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = data[idx];
        for (int i = 0; i < iterations; i++) {
            val = sinf(val) * cosf(val) + sqrtf(fabsf(val));
        }
        data[idx] = val;
    }
}

__global__ void memory_kernel(float *in, float *out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = in[idx] * 2.0f + 1.0f;
    }
}

__global__ void latency_kernel(float *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = data[idx] + 1.0f;
    }
}

/* ============================================================================
 * Benchmark Functions
 * ============================================================================ */

static void benchmark_graph_capture_overhead(void) {
    printf("%-50s ", "Graph Capture Overhead");
    fflush(stdout);

    multi_graph_manager_t *mgr = multi_graph_manager_init(0, true);
    if (!mgr) {
        printf("FAIL\n");
        return;
    }

    uint64_t tenant = multi_graph_register_tenant(mgr, "CaptureTest", NULL);

    float *d_data;
    int size = 1024;
    cudaMalloc(&d_data, size * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    const int iters = 100;
    uint64_t total_time = 0;

    for (int i = 0; i < iters; i++) {
        uint64_t start = get_time_ns();

        multi_graph_begin_capture(mgr, tenant, stream, "BenchGraph");
        latency_kernel<<<(size+255)/256, 256, 0, stream>>>(d_data, size);
        uint64_t graph_id = 0;
        multi_graph_end_capture(mgr, stream, &graph_id);

        uint64_t end = get_time_ns();
        total_time += (end - start);

        multi_graph_delete(mgr, graph_id);
    }

    double avg_us = (total_time / (double)iters) / 1000.0;

    printf("%.2f us/capture\n", avg_us);

    cudaFree(d_data);
    cudaStreamDestroy(stream);
    multi_graph_manager_shutdown(mgr);
}

static void benchmark_scheduling_latency(void) {
    printf("%-50s ", "Scheduling Latency (submit -> execute)");
    fflush(stdout);

    multi_graph_manager_t *mgr = multi_graph_manager_init(0, true);
    if (!mgr) {
        printf("FAIL\n");
        return;
    }

    uint64_t tenant = multi_graph_register_tenant(mgr, "LatencyTest", NULL);

    float *d_data;
    int size = 512;
    cudaMalloc(&d_data, size * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    /* Capture a simple graph */
    multi_graph_begin_capture(mgr, tenant, stream, "LatencyGraph");
    latency_kernel<<<(size+255)/256, 256, 0, stream>>>(d_data, size);
    uint64_t graph_id = 0;
    multi_graph_end_capture(mgr, stream, &graph_id);

    /* Measure submit -> execution latency */
    const int iters = 50;
    uint64_t total_latency = 0;

    for (int i = 0; i < iters; i++) {
        uint64_t start = get_time_ns();
        multi_graph_submit(mgr, graph_id);
        multi_graph_wait(mgr, graph_id, 5000);
        uint64_t end = get_time_ns();

        total_latency += (end - start);
    }

    double avg_us = (total_latency / (double)iters) / 1000.0;

    printf("%.2f us\n", avg_us);

    cudaFree(d_data);
    cudaStreamDestroy(stream);
    multi_graph_manager_shutdown(mgr);
}

static void benchmark_multi_tenant_fairness(void) {
    printf("%-50s ", "Multi-Tenant Fairness (3 tenants)");
    fflush(stdout);

    multi_graph_manager_t *mgr = multi_graph_manager_init(0, true);
    if (!mgr) {
        printf("FAIL\n");
        return;
    }

    /* Register 3 tenants with different priorities */
    slo_config_t high_slo = {
        .target_latency_ms = 10.0,
        .max_latency_ms = 50.0,
        .priority = 90,
    };
    slo_config_t med_slo = {
        .target_latency_ms = 50.0,
        .max_latency_ms = 100.0,
        .priority = 50,
    };
    slo_config_t low_slo = {
        .target_latency_ms = 100.0,
        .max_latency_ms = 200.0,
        .priority = 20,
    };

    uint64_t tenant_high = multi_graph_register_tenant(mgr, "HighPriority", &high_slo);
    uint64_t tenant_med = multi_graph_register_tenant(mgr, "MedPriority", &med_slo);
    uint64_t tenant_low = multi_graph_register_tenant(mgr, "LowPriority", &low_slo);

    /* Create graphs for each tenant */
    float *d_data[3];
    cudaStream_t streams[3];
    uint64_t graph_ids[3];
    uint64_t tenants[] = {tenant_high, tenant_med, tenant_low};

    int size = 2048;
    for (int i = 0; i < 3; i++) {
        cudaMalloc(&d_data[i], size * sizeof(float));
        cudaStreamCreate(&streams[i]);

        multi_graph_begin_capture(mgr, tenants[i], streams[i], "TenantGraph");
        compute_kernel<<<(size+255)/256, 256, 0, streams[i]>>>(d_data[i], size, 100 * (i+1));
        multi_graph_end_capture(mgr, streams[i], &graph_ids[i]);
    }

    /* Submit all graphs concurrently multiple times */
    const int rounds = 20;
    uint64_t latencies[3] = {0};

    for (int r = 0; r < rounds; r++) {
        /* Submit all in reverse priority order */
        for (int i = 2; i >= 0; i--) {
            uint64_t start = get_time_ns();
            multi_graph_submit(mgr, graph_ids[i]);
            multi_graph_wait(mgr, graph_ids[i], 5000);
            uint64_t end = get_time_ns();
            latencies[i] += (end - start);
        }
    }

    /* Calculate average latencies */
    double avg_high_ms = (latencies[0] / (double)rounds) / 1000000.0;
    double avg_med_ms = (latencies[1] / (double)rounds) / 1000000.0;
    double avg_low_ms = (latencies[2] / (double)rounds) / 1000000.0;

    printf("H:%.1fms M:%.1fms L:%.1fms\n", avg_high_ms, avg_med_ms, avg_low_ms);

    for (int i = 0; i < 3; i++) {
        cudaFree(d_data[i]);
        cudaStreamDestroy(streams[i]);
    }

    multi_graph_manager_shutdown(mgr);
}

static void benchmark_slo_compliance(void) {
    printf("%-50s ", "SLO Compliance Rate");
    fflush(stdout);

    multi_graph_manager_t *mgr = multi_graph_manager_init(0, true);
    if (!mgr) {
        printf("FAIL\n");
        return;
    }

    /* Register tenant with reasonable SLO */
    slo_config_t slo = {
        .target_latency_ms = 20.0,
        .max_latency_ms = 50.0,
        .priority = 70,
    };

    uint64_t tenant = multi_graph_register_tenant(mgr, "SLOTest", &slo);

    float *d_data;
    int size = 1024;
    cudaMalloc(&d_data, size * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    /* Capture moderate workload */
    multi_graph_begin_capture(mgr, tenant, stream, "SLOGraph");
    compute_kernel<<<(size+255)/256, 256, 0, stream>>>(d_data, size, 500);
    uint64_t graph_id = 0;
    multi_graph_end_capture(mgr, stream, &graph_id);

    /* Execute many times */
    const int executions = 100;
    for (int i = 0; i < executions; i++) {
        multi_graph_submit(mgr, graph_id);
        multi_graph_wait(mgr, graph_id, 5000);
    }

    /* Get statistics */
    multi_graph_stats_t stats;
    multi_graph_get_stats(mgr, &stats);

    double compliance = 100.0 * (1.0 - (double)stats.slo_violations_total / executions);

    printf("%.1f%% (%lu violations / %d executions)\n",
           compliance, stats.slo_violations_total, executions);

    cudaFree(d_data);
    cudaStreamDestroy(stream);
    multi_graph_manager_shutdown(mgr);
}

static void benchmark_throughput_scaling(void) {
    printf("%-50s ", "Throughput Scaling");
    fflush(stdout);

    multi_graph_manager_t *mgr = multi_graph_manager_init(0, true);
    if (!mgr) {
        printf("FAIL\n");
        return;
    }

    int tenant_counts[] = {1, 2, 4, 8};

    for (int tc = 0; tc < 4; tc++) {
        int num_tenants = tenant_counts[tc];
        uint64_t tenants[8];
        uint64_t graph_ids[8];
        float *d_data[8];
        cudaStream_t streams[8];

        /* Register tenants and create graphs */
        int size = 1024;
        for (int i = 0; i < num_tenants; i++) {
            char name[32];
            snprintf(name, sizeof(name), "Tenant%d", i);
            tenants[i] = multi_graph_register_tenant(mgr, name, NULL);

            cudaMalloc(&d_data[i], size * sizeof(float));
            cudaStreamCreate(&streams[i]);

            multi_graph_begin_capture(mgr, tenants[i], streams[i], "WorkGraph");
            compute_kernel<<<(size+255)/256, 256, 0, streams[i]>>>(d_data[i], size, 200);
            multi_graph_end_capture(mgr, streams[i], &graph_ids[i]);
        }

        /* Measure throughput */
        const int total_submissions = 50;
        uint64_t start = get_time_ns();

        for (int iter = 0; iter < total_submissions; iter++) {
            for (int i = 0; i < num_tenants; i++) {
                multi_graph_submit(mgr, graph_ids[i]);
            }
        }

        /* Wait for all to complete */
        for (int iter = 0; iter < total_submissions; iter++) {
            for (int i = 0; i < num_tenants; i++) {
                multi_graph_wait(mgr, graph_ids[i], 10000);
            }
        }

        uint64_t end = get_time_ns();
        double elapsed_s = (end - start) / 1000000000.0;
        double throughput = (total_submissions * num_tenants) / elapsed_s;

        printf("%.0fops/s@%dT ", throughput, num_tenants);

        /* Cleanup */
        for (int i = 0; i < num_tenants; i++) {
            cudaFree(d_data[i]);
            cudaStreamDestroy(streams[i]);
            multi_graph_unregister_tenant(mgr, tenants[i]);
        }
    }

    printf("\n");

    multi_graph_manager_shutdown(mgr);
}

static void benchmark_graph_execution_overhead(void) {
    printf("%-50s ", "Graph vs Direct Launch Overhead");
    fflush(stdout);

    multi_graph_manager_t *mgr = multi_graph_manager_init(0, true);
    if (!mgr) {
        printf("FAIL\n");
        return;
    }

    uint64_t tenant = multi_graph_register_tenant(mgr, "OverheadTest", NULL);

    float *d_data;
    int size = 512;
    cudaMalloc(&d_data, size * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    /* Measure direct launch time */
    const int iters = 100;
    uint64_t direct_time = 0;

    for (int i = 0; i < iters; i++) {
        uint64_t start = get_time_ns();
        latency_kernel<<<(size+255)/256, 256, 0, stream>>>(d_data, size);
        cudaStreamSynchronize(stream);
        uint64_t end = get_time_ns();
        direct_time += (end - start);
    }

    /* Measure graph launch time */
    multi_graph_begin_capture(mgr, tenant, stream, "OverheadGraph");
    latency_kernel<<<(size+255)/256, 256, 0, stream>>>(d_data, size);
    uint64_t graph_id = 0;
    multi_graph_end_capture(mgr, stream, &graph_id);

    uint64_t graph_time = 0;

    for (int i = 0; i < iters; i++) {
        uint64_t start = get_time_ns();
        multi_graph_submit(mgr, graph_id);
        multi_graph_wait(mgr, graph_id, 5000);
        uint64_t end = get_time_ns();
        graph_time += (end - start);
    }

    double direct_avg_us = (direct_time / (double)iters) / 1000.0;
    double graph_avg_us = (graph_time / (double)iters) / 1000.0;
    double overhead_pct = ((graph_avg_us - direct_avg_us) / direct_avg_us) * 100.0;

    printf("Direct:%.1fus Graph:%.1fus (+%.1f%%)\n",
           direct_avg_us, graph_avg_us, overhead_pct);

    cudaFree(d_data);
    cudaStreamDestroy(stream);
    multi_graph_manager_shutdown(mgr);
}

/* ============================================================================
 * Main Benchmark Runner
 * ============================================================================ */

extern "C" void run_multi_graph_benchmarks(void) {
    printf("\n=================================================\n");
    printf("Multi-Graph Multi-Tenant Manager Benchmarks\n");
    printf("=================================================\n\n");

    benchmark_graph_capture_overhead();
    benchmark_scheduling_latency();
    benchmark_graph_execution_overhead();
    benchmark_multi_tenant_fairness();
    benchmark_throughput_scaling();
    benchmark_slo_compliance();

    printf("\n");
}
