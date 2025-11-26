/*
 * GPU Virtualization Performance Evaluation Tool
 * MIG (Multi-Instance GPU) Baseline Simulator
 *
 * This module provides expected "ideal" MIG behavior values for comparison.
 * Since MIG is hardware-based, we cannot truly simulate it in software.
 * Instead, we define the expected characteristics based on NVIDIA documentation
 * and published benchmarks.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "include/benchmark.h"
#include "include/metrics.h"

/*
 * ============================================================================
 * MIG Instance Profiles (based on A100/H100)
 * ============================================================================
 */

typedef enum {
    MIG_PROFILE_1G_5GB = 0,     /* 1/7 GPU, 5GB memory */
    MIG_PROFILE_1G_10GB,        /* 1/7 GPU, 10GB memory (A100-80G) */
    MIG_PROFILE_2G_10GB,        /* 2/7 GPU, 10GB memory */
    MIG_PROFILE_3G_20GB,        /* 3/7 GPU, 20GB memory */
    MIG_PROFILE_4G_20GB,        /* 4/7 GPU, 20GB memory */
    MIG_PROFILE_7G_40GB,        /* Full GPU, 40GB memory */
    MIG_PROFILE_7G_80GB,        /* Full GPU, 80GB memory */
    MIG_PROFILE_COUNT
} mig_profile_t;

typedef struct {
    mig_profile_t profile;
    const char *name;
    int sm_count;               /* SMs allocated to instance */
    int sm_percentage;          /* % of total SMs */
    uint64_t memory_bytes;      /* Memory allocated */
    double memory_bandwidth_ratio; /* Fraction of total bandwidth */
    double l2_cache_ratio;      /* Fraction of L2 cache */
    int max_instances;          /* Max instances of this profile per GPU */
} mig_profile_spec_t;

/* A100-40GB specifications (108 SMs total) */
static const mig_profile_spec_t MIG_PROFILES_A100_40G[] = {
    {MIG_PROFILE_1G_5GB,   "1g.5gb",   14, 13,  5ULL * 1024*1024*1024, 0.14, 0.14, 7},
    {MIG_PROFILE_2G_10GB,  "2g.10gb",  28, 26, 10ULL * 1024*1024*1024, 0.28, 0.28, 3},
    {MIG_PROFILE_3G_20GB,  "3g.20gb",  42, 39, 20ULL * 1024*1024*1024, 0.43, 0.43, 2},
    {MIG_PROFILE_4G_20GB,  "4g.20gb",  56, 52, 20ULL * 1024*1024*1024, 0.57, 0.57, 1},
    {MIG_PROFILE_7G_40GB,  "7g.40gb", 108,100, 40ULL * 1024*1024*1024, 1.00, 1.00, 1},
};

/*
 * ============================================================================
 * MIG Expected Performance Characteristics
 * ============================================================================
 */

typedef struct {
    /* Overhead characteristics (MIG has minimal software overhead) */
    double kernel_launch_overhead_us;   /* ~5us typical */
    double mem_alloc_overhead_us;       /* Similar to native */
    double mem_free_overhead_us;
    double context_overhead_us;
    double api_overhead_ns;             /* 0 - no interception */

    /* Isolation characteristics (MIG provides hardware isolation) */
    double memory_isolation_percent;    /* 100% - hardware enforced */
    double sm_isolation_percent;        /* 100% - hardware assigned */
    double l2_isolation_percent;        /* 100% - partitioned */
    double bandwidth_isolation_percent; /* 100% - dedicated controllers */
    double fault_isolation;             /* 1.0 - hardware isolated */

    /* QoS characteristics */
    double qos_variance;                /* Very low - hardware guarantees */
    double fairness_index;              /* ~0.98 - near perfect */
    double noisy_neighbor_impact;       /* ~2% - only PCIe/power shared */

    /* Performance scaling */
    double throughput_per_sm;           /* Linear with SM count */
    double memory_bandwidth_per_slice;  /* Linear with slices */

} mig_expected_perf_t;

static const mig_expected_perf_t MIG_EXPECTED_PERF = {
    /* Overhead - MIG is hardware, minimal software overhead */
    .kernel_launch_overhead_us = 5.0,
    .mem_alloc_overhead_us = 50.0,
    .mem_free_overhead_us = 20.0,
    .context_overhead_us = 100.0,
    .api_overhead_ns = 0.0,           /* No software interception */

    /* Isolation - MIG provides hardware isolation */
    .memory_isolation_percent = 100.0,
    .sm_isolation_percent = 100.0,
    .l2_isolation_percent = 100.0,
    .bandwidth_isolation_percent = 100.0,
    .fault_isolation = 1.0,

    /* QoS - MIG provides strong guarantees */
    .qos_variance = 0.03,             /* ~3% variance */
    .fairness_index = 0.98,           /* Near-perfect fairness */
    .noisy_neighbor_impact = 2.0,     /* ~2% from PCIe/power sharing */

    /* Performance */
    .throughput_per_sm = 1.0,         /* Linear scaling */
    .memory_bandwidth_per_slice = 1.0,
};

/*
 * ============================================================================
 * MIG Baseline Value Generator
 * ============================================================================
 */

/**
 * Get expected MIG value for a specific metric
 */
double mig_get_expected_value(const char *metric_id, const mig_profile_spec_t *profile) {
    /* Overhead metrics */
    if (strcmp(metric_id, METRIC_OH_KERNEL_LAUNCH_LATENCY) == 0) {
        return MIG_EXPECTED_PERF.kernel_launch_overhead_us;
    }
    if (strcmp(metric_id, METRIC_OH_MEMORY_ALLOC_LATENCY) == 0) {
        return MIG_EXPECTED_PERF.mem_alloc_overhead_us;
    }
    if (strcmp(metric_id, METRIC_OH_MEMORY_FREE_LATENCY) == 0) {
        return MIG_EXPECTED_PERF.mem_free_overhead_us;
    }
    if (strcmp(metric_id, METRIC_OH_CONTEXT_CREATION) == 0) {
        return MIG_EXPECTED_PERF.context_overhead_us;
    }
    if (strcmp(metric_id, METRIC_OH_API_INTERCEPTION) == 0) {
        return MIG_EXPECTED_PERF.api_overhead_ns;
    }
    if (strcmp(metric_id, METRIC_OH_LOCK_CONTENTION) == 0) {
        return 0.0;  /* No software locks */
    }
    if (strcmp(metric_id, METRIC_OH_MEMORY_TRACKING) == 0) {
        return 0.0;  /* No software tracking */
    }
    if (strcmp(metric_id, METRIC_OH_RATE_LIMITER) == 0) {
        return 0.0;  /* No software rate limiting */
    }
    if (strcmp(metric_id, METRIC_OH_NVML_POLLING) == 0) {
        return 0.0;  /* No polling overhead */
    }
    if (strcmp(metric_id, METRIC_OH_TOTAL_THROUGHPUT) == 0) {
        return 2.0;  /* ~2% overhead from instance management */
    }

    /* Isolation metrics */
    if (strcmp(metric_id, METRIC_IS_MEMORY_LIMIT_ACCURACY) == 0) {
        return MIG_EXPECTED_PERF.memory_isolation_percent;
    }
    if (strcmp(metric_id, METRIC_IS_MEMORY_LIMIT_ENFORCEMENT) == 0) {
        return 1.0;  /* Instant - hardware enforced */
    }
    if (strcmp(metric_id, METRIC_IS_SM_UTIL_ACCURACY) == 0) {
        return MIG_EXPECTED_PERF.sm_isolation_percent;
    }
    if (strcmp(metric_id, METRIC_IS_SM_LIMIT_RESPONSE) == 0) {
        return 0.0;  /* Instant - static assignment */
    }
    if (strcmp(metric_id, METRIC_IS_CROSS_TENANT_MEMORY) == 0) {
        return 1.0;  /* Full hardware isolation */
    }
    if (strcmp(metric_id, METRIC_IS_CROSS_TENANT_COMPUTE) == 0) {
        return 100.0 - MIG_EXPECTED_PERF.noisy_neighbor_impact;
    }
    if (strcmp(metric_id, METRIC_IS_QOS_CONSISTENCY) == 0) {
        return MIG_EXPECTED_PERF.qos_variance;
    }
    if (strcmp(metric_id, METRIC_IS_FAIRNESS_INDEX) == 0) {
        return MIG_EXPECTED_PERF.fairness_index;
    }
    if (strcmp(metric_id, METRIC_IS_NOISY_NEIGHBOR) == 0) {
        return MIG_EXPECTED_PERF.noisy_neighbor_impact;
    }
    if (strcmp(metric_id, METRIC_IS_FAULT_ISOLATION) == 0) {
        return MIG_EXPECTED_PERF.fault_isolation;
    }

    /* LLM metrics - scaled by profile */
    if (strcmp(metric_id, METRIC_LLM_ATTENTION_THROUGHPUT) == 0) {
        if (profile != NULL) {
            /* TFLOPS scales with SM count */
            /* A100 full GPU ~312 TFLOPS FP16, scale by SM percentage */
            return 312.0 * (profile->sm_percentage / 100.0);
        }
        return 312.0;  /* Full GPU */
    }
    if (strcmp(metric_id, METRIC_LLM_KV_CACHE_ALLOC) == 0) {
        return 100000.0;  /* Near-native allocation speed */
    }
    if (strcmp(metric_id, METRIC_LLM_BATCH_SIZE_SCALING) == 0) {
        return 0.95;  /* ~95% linear scaling */
    }
    if (strcmp(metric_id, METRIC_LLM_TOKEN_LATENCY) == 0) {
        /* Depends on model, use placeholder */
        return 10.0;  /* 10ms per token for reference model */
    }
    if (strcmp(metric_id, METRIC_LLM_MEMORY_POOL) == 0) {
        return 5.0;  /* ~5% pool overhead */
    }
    if (strcmp(metric_id, METRIC_LLM_MULTI_STREAM) == 0) {
        return 95.0;  /* ~95% multi-stream efficiency */
    }
    if (strcmp(metric_id, METRIC_LLM_LARGE_TENSOR) == 0) {
        return 50.0;  /* ~50ms for large tensor */
    }
    if (strcmp(metric_id, METRIC_LLM_MIXED_PRECISION) == 0) {
        return 1.9;  /* ~1.9x FP16 speedup */
    }
    if (strcmp(metric_id, METRIC_LLM_DYNAMIC_BATCHING) == 0) {
        return 0.05;  /* ~5% variance */
    }
    if (strcmp(metric_id, METRIC_LLM_MULTI_GPU_SCALING) == 0) {
        return 1.0;  /* MIG is single-instance, doesn't span */
    }

    return 0.0;  /* Unknown metric */
}

/*
 * ============================================================================
 * MIG Comparison Score Calculation
 * ============================================================================
 */

/**
 * Calculate how close a measured value is to MIG expected value
 * Returns score from 0-100 (100 = matches MIG perfectly)
 */
double mig_comparison_score(const char *metric_id, double measured_value) {
    const metric_definition_t *def = metrics_get_definition(metric_id);
    if (def == NULL) return 0.0;

    double mig_expected = mig_get_expected_value(metric_id, NULL);

    /* Handle special cases */
    if (mig_expected == 0.0 && measured_value == 0.0) {
        return 100.0;  /* Both zero = perfect match */
    }
    if (mig_expected == 0.0) {
        /* MIG has zero overhead, any overhead is bad */
        /* Score decreases with increasing measured value */
        double penalty = fmin(measured_value / 100.0, 1.0);
        return 100.0 * (1.0 - penalty);
    }

    /* Calculate relative difference */
    double relative_diff = fabs(measured_value - mig_expected) / mig_expected;

    /* Score based on whether higher is better */
    if (def->higher_is_better) {
        /* For "higher is better" metrics, being above MIG is OK */
        if (measured_value >= mig_expected) {
            return 100.0;
        }
        /* Being below MIG penalizes score */
        return 100.0 * (1.0 - fmin(relative_diff, 1.0));
    } else {
        /* For "lower is better" metrics, being below MIG is OK */
        if (measured_value <= mig_expected) {
            return 100.0;
        }
        /* Being above MIG penalizes score */
        return 100.0 * (1.0 - fmin(relative_diff, 1.0));
    }
}

/*
 * ============================================================================
 * MIG Baseline Report Generation
 * ============================================================================
 */

typedef struct {
    char metric_id[16];
    double measured_value;
    double mig_expected;
    double mig_score;
    const char *interpretation;
} mig_comparison_entry_t;

/**
 * Generate MIG comparison report for a set of results
 */
int mig_generate_comparison_report(
    const metric_result_t *results,
    int result_count,
    mig_comparison_entry_t *comparison,
    int *comparison_count
) {
    *comparison_count = 0;

    for (int i = 0; i < result_count; i++) {
        if (!results[i].valid) continue;

        mig_comparison_entry_t *entry = &comparison[*comparison_count];

        strcpy(entry->metric_id, results[i].metric_id);
        entry->measured_value = results[i].value;
        entry->mig_expected = mig_get_expected_value(results[i].metric_id, NULL);
        entry->mig_score = mig_comparison_score(results[i].metric_id, results[i].value);

        /* Generate interpretation */
        if (entry->mig_score >= 90.0) {
            entry->interpretation = "Excellent - Near MIG performance";
        } else if (entry->mig_score >= 70.0) {
            entry->interpretation = "Good - Acceptable for most workloads";
        } else if (entry->mig_score >= 50.0) {
            entry->interpretation = "Fair - May impact sensitive workloads";
        } else {
            entry->interpretation = "Poor - Significant gap from MIG";
        }

        (*comparison_count)++;
    }

    return 0;
}

/**
 * Calculate overall MIG similarity score
 */
double mig_calculate_overall_score(
    const metric_result_t *results,
    int result_count,
    metric_category_t category
) {
    double total_score = 0.0;
    int valid_count = 0;

    for (int i = 0; i < result_count; i++) {
        if (!results[i].valid) continue;

        const metric_definition_t *def = metrics_get_definition(results[i].metric_id);
        if (def == NULL) continue;

        /* Filter by category if specified */
        if (category != METRIC_CATEGORY_COUNT && def->category != category) {
            continue;
        }

        double score = mig_comparison_score(results[i].metric_id, results[i].value);
        total_score += score;
        valid_count++;
    }

    if (valid_count == 0) return 0.0;
    return total_score / valid_count;
}

/*
 * ============================================================================
 * MIG Limitations Documentation
 * ============================================================================
 */

typedef struct {
    const char *aspect;
    const char *mig_behavior;
    const char *software_virt_behavior;
    const char *gap_closable;
} mig_gap_analysis_t;

static const mig_gap_analysis_t MIG_GAP_ANALYSIS[] = {
    {
        .aspect = "Memory Isolation",
        .mig_behavior = "Hardware-enforced memory partitioning. Each instance has dedicated "
                        "memory controllers and DRAM banks.",
        .software_virt_behavior = "Software tracking via allocation interception. Relies on "
                                   "CUDA driver's address space isolation.",
        .gap_closable = "PARTIAL - Can enforce limits, cannot prevent side-channel leakage"
    },
    {
        .aspect = "L2 Cache Isolation",
        .mig_behavior = "Hardware-partitioned L2 cache slices assigned per instance.",
        .software_virt_behavior = "Shared L2 cache. No software control over cache allocation.",
        .gap_closable = "NO - Requires hardware support"
    },
    {
        .aspect = "SM/Compute Isolation",
        .mig_behavior = "Fixed SM assignment per instance. Hardware scheduler enforces.",
        .software_virt_behavior = "Token bucket rate limiting on kernel launches. Time-shared "
                                   "access to all SMs.",
        .gap_closable = "PARTIAL - Can achieve average utilization limit, cannot provide "
                        "dedicated SMs"
    },
    {
        .aspect = "Memory Bandwidth",
        .mig_behavior = "Dedicated memory controllers per instance. Guaranteed bandwidth.",
        .software_virt_behavior = "Shared bandwidth. No isolation mechanism.",
        .gap_closable = "NO - Requires hardware support"
    },
    {
        .aspect = "Fault Isolation",
        .mig_behavior = "Hardware-level fault containment. GPU errors in one instance don't "
                        "affect others.",
        .software_virt_behavior = "Process-level isolation. Some errors can propagate through "
                                   "shared driver state.",
        .gap_closable = "PARTIAL - Process isolation helps, but not as robust as hardware"
    },
    {
        .aspect = "QoS Guarantees",
        .mig_behavior = "Hardware enforces strict resource partitioning. Predictable latency "
                        "and throughput.",
        .software_virt_behavior = "Best-effort enforcement with feedback loops. Higher variance.",
        .gap_closable = "PARTIAL - Can improve with better algorithms, but inherent variance"
    },
    {
        .aspect = "Reconfiguration",
        .mig_behavior = "Requires GPU reset. Static configuration during use.",
        .software_virt_behavior = "Dynamic reconfiguration without reset.",
        .gap_closable = "N/A - Software has advantage here"
    },
    {
        .aspect = "Hardware Support",
        .mig_behavior = "Ampere+ GPUs only (A100, H100, etc.)",
        .software_virt_behavior = "Any CUDA GPU",
        .gap_closable = "N/A - Software has advantage here"
    },
    {
        .aspect = "Granularity",
        .mig_behavior = "Fixed profiles (1/7, 2/7, 3/7, etc.)",
        .software_virt_behavior = "Arbitrary percentage",
        .gap_closable = "N/A - Software has advantage here"
    },
};

#define MIG_GAP_ANALYSIS_COUNT (sizeof(MIG_GAP_ANALYSIS) / sizeof(MIG_GAP_ANALYSIS[0]))

/**
 * Print MIG gap analysis
 */
void mig_print_gap_analysis(void) {
    printf("\n");
    printf("================================================================================\n");
    printf("MIG vs Software Virtualization Gap Analysis\n");
    printf("================================================================================\n\n");

    for (int i = 0; i < MIG_GAP_ANALYSIS_COUNT; i++) {
        const mig_gap_analysis_t *gap = &MIG_GAP_ANALYSIS[i];
        printf("--- %s ---\n", gap->aspect);
        printf("MIG:      %s\n", gap->mig_behavior);
        printf("Software: %s\n", gap->software_virt_behavior);
        printf("Gap:      %s\n\n", gap->gap_closable);
    }
}

/*
 * ============================================================================
 * MIG Simulation System Interface
 * ============================================================================
 */

/**
 * Initialize MIG simulator with specified profile
 */
int mig_simulator_init(mig_profile_t profile) {
    LOG_INFO("MIG Simulator initialized with profile: %s",
             MIG_PROFILES_A100_40G[profile].name);

    /* MIG simulator doesn't actually run anything -
     * it just provides expected values for comparison */

    return 0;
}

/**
 * "Run" MIG benchmark - returns expected values as results
 */
int mig_simulator_run_benchmark(
    const char *metric_id,
    metric_result_t *result
) {
    strcpy(result->metric_id, metric_id);
    result->device_id = 0;

    /* Get expected MIG value */
    result->value = mig_get_expected_value(metric_id, NULL);

    /* Generate synthetic statistics (MIG has low variance) */
    result->stats.mean = result->value;
    result->stats.median = result->value;
    result->stats.std_dev = result->value * 0.03;  /* 3% std dev */
    result->stats.min = result->value * 0.95;
    result->stats.max = result->value * 1.05;
    result->stats.p50 = result->value;
    result->stats.p90 = result->value * 1.02;
    result->stats.p95 = result->value * 1.03;
    result->stats.p99 = result->value * 1.05;
    result->stats.p999 = result->value * 1.05;
    result->stats.count = 100;

    /* No raw values for simulated results */
    result->raw_values = NULL;
    result->raw_count = 0;

    result->timestamp_ns = timing_get_ns();
    result->valid = true;

    return 0;
}

/**
 * Generate full MIG baseline results for comparison
 */
int mig_simulator_generate_baseline(benchmark_result_t *results) {
    strcpy(results->benchmark_name, "MIG Ideal Baseline");
    strcpy(results->system_name, "mig-ideal");

    results->results = (metric_result_t*)calloc(TOTAL_METRIC_COUNT, sizeof(metric_result_t));
    if (results->results == NULL) {
        strcpy(results->error_msg, "Failed to allocate results");
        results->success = false;
        return -1;
    }

    results->result_count = 0;

    /* Generate expected values for all metrics */
    for (int i = 0; i < TOTAL_METRIC_COUNT; i++) {
        const metric_definition_t *def = &METRIC_DEFINITIONS[i];
        mig_simulator_run_benchmark(def->id, &results->results[results->result_count]);
        results->result_count++;
    }

    results->total_time_ms = 0.0;  /* Simulation is instant */
    results->success = true;

    LOG_INFO("MIG baseline generated: %d metrics", results->result_count);

    return 0;
}
