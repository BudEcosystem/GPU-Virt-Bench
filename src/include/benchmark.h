/*
 * GPU Virtualization Performance Evaluation Tool
 * Core benchmark definitions and structures
 */

#ifndef GPU_VIRT_BENCH_BENCHMARK_H
#define GPU_VIRT_BENCH_BENCHMARK_H

#include <stdint.h>
#include <stdbool.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * ============================================================================
 * Version and Constants
 * ============================================================================
 */

#define BENCH_VERSION_MAJOR 1
#define BENCH_VERSION_MINOR 0
#define BENCH_VERSION_PATCH 0

#define MAX_DEVICES 16
#define MAX_SYSTEMS 8
#define MAX_BENCHMARKS 64
#define MAX_NAME_LENGTH 128
#define MAX_PATH_LENGTH 512
#define MAX_ITERATIONS 10000
#define DEFAULT_ITERATIONS 100
#define DEFAULT_WARMUP 10

/*
 * ============================================================================
 * Timing Utilities
 * ============================================================================
 */

typedef struct {
    uint64_t start_ns;
    uint64_t end_ns;
    double elapsed_us;
    double elapsed_ms;
} timing_result_t;

typedef struct {
    double min;
    double max;
    double mean;
    double median;
    double std_dev;
    double p50;      /* 50th percentile */
    double p90;      /* 90th percentile */
    double p95;      /* 95th percentile */
    double p99;      /* 99th percentile */
    double p999;     /* 99.9th percentile */
    uint64_t count;
} statistics_t;

/*
 * ============================================================================
 * Metric Definitions
 * ============================================================================
 */

typedef enum {
    METRIC_CATEGORY_OVERHEAD = 0,
    METRIC_CATEGORY_ISOLATION = 1,
    METRIC_CATEGORY_LLM = 2,
    METRIC_CATEGORY_COUNT
} metric_category_t;

typedef enum {
    METRIC_UNIT_NANOSECONDS = 0,
    METRIC_UNIT_MICROSECONDS,
    METRIC_UNIT_MILLISECONDS,
    METRIC_UNIT_SECONDS,
    METRIC_UNIT_PERCENTAGE,
    METRIC_UNIT_BOOLEAN,
    METRIC_UNIT_RATIO,
    METRIC_UNIT_THROUGHPUT,     /* ops/sec */
    METRIC_UNIT_BANDWIDTH,      /* GB/s */
    METRIC_UNIT_FLOPS,          /* TFLOPS */
    METRIC_UNIT_TOKENS_PER_SEC,
    METRIC_UNIT_ALLOCS_PER_SEC,
    METRIC_UNIT_COUNT
} metric_unit_t;

typedef struct {
    char id[16];                /* e.g., "OH-001" */
    char name[MAX_NAME_LENGTH];
    char description[256];
    metric_category_t category;
    metric_unit_t unit;
    bool higher_is_better;      /* true = higher values are better */
} metric_definition_t;

/*
 * ============================================================================
 * Benchmark Result Structures
 * ============================================================================
 */

typedef struct {
    char metric_id[16];
    char name[MAX_NAME_LENGTH];      /* Metric name */
    char unit[32];                   /* Unit string */
    double value;
    statistics_t stats;
    double *raw_values;              /* Raw measurement array */
    uint64_t raw_count;
    uint64_t timestamp_ns;
    int device_id;
    bool valid;
    bool success;                    /* Alias for valid */
    char error_msg[256];
    char error_message[256];         /* Alias for error_msg */
    double mig_expected;             /* Expected MIG value */
    double mig_gap_percent;          /* Gap from MIG baseline */
    double normalized_score;         /* Score 0-1 vs MIG */
} metric_result_t;

typedef struct {
    int warmup_iterations;
    int benchmark_iterations;
    int num_processes;
    size_t memory_limit_mb;
    int compute_limit_percent;
} benchmark_config_inner_t;

typedef struct {
    char benchmark_name[MAX_NAME_LENGTH];
    char system_name[MAX_NAME_LENGTH];
    char system_version[64];
    metric_result_t *results;
    metric_result_t *metrics;        /* Alias for results */
    int result_count;
    int num_metrics;                 /* Alias for result_count */
    double total_time_ms;
    double total_duration_seconds;   /* Alias */
    bool success;
    char error_msg[256];
    benchmark_config_inner_t config;
    double overhead_score;
    double isolation_score;
    double llm_score;
    double overall_score;
    double mig_parity_percent;
} benchmark_result_t;

/*
 * ============================================================================
 * System Configuration
 * ============================================================================
 */

typedef enum {
    VIRT_SYSTEM_NATIVE = 0,     /* No virtualization */
    VIRT_SYSTEM_HAMI,           /* HaMi-core original */
    VIRT_SYSTEM_FCSP,           /* Our improved version */
    VIRT_SYSTEM_MIG_IDEAL,      /* Simulated ideal MIG */
    VIRT_SYSTEM_COUNT
} virt_system_type_t;

typedef struct {
    virt_system_type_t type;
    char name[MAX_NAME_LENGTH];
    char library_path[MAX_PATH_LENGTH];
    bool enabled;

    /* Resource limits */
    uint64_t memory_limit_bytes;
    int sm_limit_percent;       /* 0-100, 0 = no limit */

    /* System-specific config */
    char shared_region_path[MAX_PATH_LENGTH];
    bool utilization_switch;
    int priority;

    /* MIG simulation parameters (for MIG_IDEAL) */
    int mig_instance_count;
    int mig_sm_per_instance;
    uint64_t mig_memory_per_instance;
    double mig_l2_cache_ratio;
    double mig_bandwidth_ratio;
} system_config_t;

/*
 * ============================================================================
 * Benchmark Configuration
 * ============================================================================
 */

typedef struct {
    int iterations;
    int warmup_iterations;
    int timeout_seconds;
    bool verbose;
    bool save_raw_data;
    char output_dir[MAX_PATH_LENGTH];
} benchmark_options_t;

typedef struct {
    char name[MAX_NAME_LENGTH];
    char config_file[MAX_PATH_LENGTH];

    benchmark_options_t options;

    system_config_t systems[MAX_SYSTEMS];
    int system_count;

    /* Which metrics to run */
    bool overhead_enabled;
    bool isolation_enabled;
    bool llm_enabled;

    char enabled_metrics[MAX_BENCHMARKS][16];
    int enabled_metric_count;

    /* Multi-tenant config */
    bool multi_tenant_mode;
    int tenant_count;

    /* Device selection */
    int device_ids[MAX_DEVICES];
    int device_count;
} benchmark_config_t;

/*
 * ============================================================================
 * MIG Baseline Simulation
 * ============================================================================
 */

/* Expected MIG behaviors for comparison */
typedef struct {
    /* Memory isolation: should be perfect (100%) */
    double expected_memory_isolation;

    /* SM isolation: hardware enforced */
    double expected_sm_isolation;

    /* L2 cache isolation: hardware partitioned */
    double expected_l2_isolation;

    /* Memory bandwidth isolation */
    double expected_bandwidth_isolation;

    /* Typical overheads */
    double typical_kernel_launch_overhead_us;
    double typical_allocation_overhead_us;

    /* QoS guarantees */
    double expected_qos_variance;       /* Should be very low */
    double expected_fairness_index;     /* Should be ~1.0 */

    /* Reconfiguration (MIG requires GPU reset) */
    double reconfiguration_time_ms;

} mig_baseline_t;

/* Default MIG baseline values based on NVIDIA documentation */
static const mig_baseline_t MIG_BASELINE_DEFAULT = {
    .expected_memory_isolation = 100.0,
    .expected_sm_isolation = 100.0,
    .expected_l2_isolation = 100.0,
    .expected_bandwidth_isolation = 100.0,
    .typical_kernel_launch_overhead_us = 5.0,
    .typical_allocation_overhead_us = 50.0,
    .expected_qos_variance = 0.05,      /* 5% variance */
    .expected_fairness_index = 0.98,
    .reconfiguration_time_ms = 30000.0, /* ~30 seconds for GPU reset */
};

/*
 * ============================================================================
 * Overhead Benchmark Parameters
 * ============================================================================
 */

typedef struct {
    /* Kernel launch tests */
    int kernel_launch_iterations;
    int kernel_sizes[8];        /* Different grid/block configurations */
    int kernel_size_count;

    /* Memory allocation tests */
    size_t allocation_sizes[16]; /* Different allocation sizes to test */
    int allocation_size_count;
    int allocations_per_size;

    /* Contention tests */
    int contention_thread_count;
    int contention_iterations;

} overhead_params_t;

static const overhead_params_t OVERHEAD_PARAMS_DEFAULT = {
    .kernel_launch_iterations = 1000,
    .kernel_sizes = {1, 32, 128, 256, 512, 1024, 4096, 16384},
    .kernel_size_count = 8,
    .allocation_sizes = {
        1024,           /* 1 KB */
        4096,           /* 4 KB */
        65536,          /* 64 KB */
        262144,         /* 256 KB */
        1048576,        /* 1 MB */
        4194304,        /* 4 MB */
        16777216,       /* 16 MB */
        67108864,       /* 64 MB */
        268435456,      /* 256 MB */
        536870912,      /* 512 MB */
        1073741824,     /* 1 GB */
    },
    .allocation_size_count = 11,
    .allocations_per_size = 100,
    .contention_thread_count = 8,
    .contention_iterations = 100,
};

/*
 * ============================================================================
 * Isolation Benchmark Parameters
 * ============================================================================
 */

typedef struct {
    /* Memory isolation tests */
    double memory_limit_percentages[5];  /* Test at different % of limit */
    int memory_limit_test_count;

    /* SM isolation tests */
    int sm_limit_values[5];     /* SM limits to test (%) */
    int sm_limit_test_count;

    /* Multi-tenant tests */
    int tenant_configs[4][2];   /* [tenant_count][memory%, sm%] */
    int tenant_config_count;

    /* Noisy neighbor simulation */
    int noisy_neighbor_intensity_levels;
    double noisy_memory_pressure_ratio;
    double noisy_compute_pressure_ratio;

} isolation_params_t;

static const isolation_params_t ISOLATION_PARAMS_DEFAULT = {
    .memory_limit_percentages = {0.5, 0.75, 0.9, 0.95, 1.0},
    .memory_limit_test_count = 5,
    .sm_limit_values = {25, 50, 75, 90, 100},
    .sm_limit_test_count = 5,
    .tenant_configs = {
        {2, 50},    /* 2 tenants, 50% each */
        {4, 25},    /* 4 tenants, 25% each */
        {8, 12},    /* 8 tenants, ~12% each */
        {10, 10},   /* 10 tenants, 10% each */
    },
    .tenant_config_count = 4,
    .noisy_neighbor_intensity_levels = 5,
    .noisy_memory_pressure_ratio = 0.9,
    .noisy_compute_pressure_ratio = 0.95,
};

/*
 * ============================================================================
 * LLM Benchmark Parameters
 * ============================================================================
 */

typedef struct {
    /* Model simulation parameters */
    int hidden_sizes[4];        /* e.g., 768, 1024, 2048, 4096 */
    int hidden_size_count;
    int sequence_lengths[4];    /* e.g., 128, 512, 1024, 2048 */
    int seq_length_count;
    int batch_sizes[6];         /* e.g., 1, 2, 4, 8, 16, 32 */
    int batch_size_count;
    int num_attention_heads[3]; /* e.g., 12, 16, 32 */
    int head_count;

    /* KV cache parameters */
    size_t kv_cache_page_size;
    int kv_cache_growth_iterations;

    /* Multi-stream (pipeline parallel) */
    int pipeline_stages[3];     /* e.g., 2, 4, 8 stages */
    int pipeline_stage_count;

    /* Tensor parallel */
    int tensor_parallel_ways[3]; /* e.g., 2, 4, 8 */
    int tp_count;

} llm_params_t;

static const llm_params_t LLM_PARAMS_DEFAULT = {
    .hidden_sizes = {768, 1024, 2048, 4096},
    .hidden_size_count = 4,
    .sequence_lengths = {128, 512, 1024, 2048},
    .seq_length_count = 4,
    .batch_sizes = {1, 2, 4, 8, 16, 32},
    .batch_size_count = 6,
    .num_attention_heads = {12, 16, 32},
    .head_count = 3,
    .kv_cache_page_size = 2097152,  /* 2 MB */
    .kv_cache_growth_iterations = 100,
    .pipeline_stages = {2, 4, 8},
    .pipeline_stage_count = 3,
    .tensor_parallel_ways = {2, 4, 8},
    .tp_count = 3,
};

/*
 * ============================================================================
 * Function Declarations
 * ============================================================================
 */

/* Initialization and cleanup */
int bench_init(benchmark_config_t *config);
void bench_cleanup(void);

/* Configuration */
int bench_load_config(const char *config_file, benchmark_config_t *config);
int bench_save_config(const char *config_file, const benchmark_config_t *config);
void bench_set_defaults(benchmark_config_t *config);

/* System management */
int bench_activate_system(virt_system_type_t system);
int bench_deactivate_system(void);
const char* bench_get_system_name(virt_system_type_t system);

/* Benchmark execution */
int bench_run_overhead(const benchmark_config_t *config, benchmark_result_t *results);
int bench_run_isolation(const benchmark_config_t *config, benchmark_result_t *results);
int bench_run_llm(const benchmark_config_t *config, benchmark_result_t *results);
int bench_run_all(const benchmark_config_t *config, benchmark_result_t **results, int *result_count);

/* New comprehensive benchmark categories */
/* Note: These use a simplified bench_config/bench_result interface for compatibility */

/* Simplified config/result types for new benchmark categories */
typedef struct {
    int iterations;
    int warmup_iterations;
    int verbose;
} bench_config_t;

typedef struct {
    char metric_id[16];
    char name[128];
    char unit[32];
    double value;
    double stddev;
    int success;
    int iterations;
    char details[256];
    char error_message[256];
} bench_result_t;

/* Bandwidth metrics - from bandwidth.cu */
void bench_run_bandwidth(bench_config_t *config, bench_result_t *results, int *count);

/* Cache metrics - from cache.cu */
void bench_run_cache(bench_config_t *config, bench_result_t *results, int *count);

/* PCIe metrics - from pcie.cu */
void bench_run_pcie(bench_config_t *config, bench_result_t *results, int *count);

/* NCCL/P2P metrics - from nccl.cu */
void bench_run_nccl(bench_config_t *config, bench_result_t *results, int *count);

/* Scheduling metrics - from scheduling.cu */
void bench_run_scheduling(bench_config_t *config, bench_result_t *results, int *count);

/* Fragmentation metrics - from fragmentation.cu */
void bench_run_fragmentation(bench_config_t *config, bench_result_t *results, int *count);

/* Error recovery metrics - from error.cu */
void bench_run_error(bench_config_t *config, bench_result_t *results, int *count);

/* Paper-inspired feature metrics - from paper_features.cu */
void bench_run_paper_features(bench_config_t *config, metric_result_t *results, int *count);

/* FCSP advanced feature metrics - from fcsp_features.cu */
void bench_run_fcsp_features(bench_config_t *config, metric_result_t *results, int *count);

/* Individual metric benchmarks */
int bench_kernel_launch_latency(metric_result_t *result);
int bench_memory_allocation_latency(metric_result_t *result);
int bench_memory_free_latency(metric_result_t *result);
int bench_context_creation_overhead(metric_result_t *result);
int bench_api_interception_overhead(metric_result_t *result);
int bench_lock_contention(metric_result_t *result);
int bench_memory_tracking_overhead(metric_result_t *result);
int bench_rate_limiter_overhead(metric_result_t *result);
int bench_nvml_polling_overhead(metric_result_t *result);
int bench_total_throughput(metric_result_t *result);

int bench_memory_limit_accuracy(metric_result_t *result);
int bench_memory_limit_enforcement(metric_result_t *result);
int bench_sm_utilization_accuracy(metric_result_t *result);
int bench_sm_limit_response_time(metric_result_t *result);
int bench_cross_tenant_memory_isolation(metric_result_t *result);
int bench_cross_tenant_compute_isolation(metric_result_t *result);
int bench_qos_consistency(metric_result_t *result);
int bench_fairness_index(metric_result_t *result);
int bench_noisy_neighbor_impact(metric_result_t *result);
int bench_fault_isolation(metric_result_t *result);

int bench_attention_kernel_throughput(metric_result_t *result);
int bench_kv_cache_allocation_speed(metric_result_t *result);
int bench_batch_size_scaling(metric_result_t *result);
int bench_token_generation_latency(metric_result_t *result);
int bench_memory_pool_efficiency(metric_result_t *result);
int bench_multi_stream_performance(metric_result_t *result);
int bench_large_tensor_allocation(metric_result_t *result);
int bench_mixed_precision_support(metric_result_t *result);
int bench_dynamic_batching_impact(metric_result_t *result);
int bench_multi_gpu_scaling(metric_result_t *result);

/* MIG baseline comparison */
int bench_compare_to_mig(const benchmark_result_t *result, const mig_baseline_t *baseline,
                         double *score);

/* Worker dispatch for multi-process tests */
int dispatch_worker(const char *test_id, const char *args);

/* Statistics and analysis */
void stats_calculate(const double *values, uint64_t count, statistics_t *stats);
double stats_jains_fairness(const double *values, int count);
double stats_coefficient_of_variation(const statistics_t *stats);

/* Timing utilities */
uint64_t timing_get_ns(void);
void timing_start(timing_result_t *t);
void timing_stop(timing_result_t *t);
void timing_cuda_sync_start(timing_result_t *t, cudaStream_t stream);
void timing_cuda_sync_stop(timing_result_t *t, cudaStream_t stream);

/* Result management */
int results_save(const char *output_dir, const benchmark_result_t *results, int count);
int results_save_json(const char *filepath, const benchmark_result_t *results, int count);
int results_save_csv(const char *filepath, const benchmark_result_t *results, int count);
int results_generate_report(const char *output_dir, const benchmark_result_t *results, int count);
void results_print_summary(const benchmark_result_t *results, int count);
void results_free(benchmark_result_t *results, int count);

/* Comparison utilities */
int compare_systems(const benchmark_result_t *baseline, const benchmark_result_t *comparison,
                    double *improvement_percent);

/* Logging */
void bench_log(int level, const char *fmt, ...);
#define LOG_DEBUG(fmt, ...) bench_log(0, fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...)  bench_log(1, fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...)  bench_log(2, fmt, ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) bench_log(3, fmt, ##__VA_ARGS__)

#ifdef __cplusplus
}
#endif

#endif /* GPU_VIRT_BENCH_BENCHMARK_H */
