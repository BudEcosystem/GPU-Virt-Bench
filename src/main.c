/*
 * GPU Virtualization Performance Evaluation Tool
 * Main entry point - Simplified version
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <getopt.h>
#include <signal.h>
#include <unistd.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <cuda_runtime.h>

#include "include/benchmark.h"
#include "include/metrics.h"

/* External function from isolation.cu for worker dispatch */
extern int dispatch_worker(const char *test_id, const char *args);

/* Global interrupt flag */
static volatile int g_interrupted = 0;

/* Stub implementations for missing functions */
void bench_log(int level, const char *fmt, ...) {
    (void)level;
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    printf("\n");
}

/* launch_worker and wait_for_worker are in process_stub.c */

const metric_definition_t* metrics_get_definition(const char *metric_id) {
    for (int i = 0; i < TOTAL_METRIC_COUNT; i++) {
        if (strcmp(METRIC_DEFINITIONS[i].id, metric_id) == 0) {
            return &METRIC_DEFINITIONS[i];
        }
    }
    return NULL;
}

static void signal_handler(int sig) {
    (void)sig;
    g_interrupted = 1;
    printf("\nInterrupted. Cleaning up...\n");
}

/* CLI options */
typedef struct {
    char system_name[64];
    char output_dir[256];
    char config_file[256];
    int iterations;
    int warmup;
    bool verbose;
    bool run_overhead;
    bool run_isolation;
    bool run_llm;
    bool run_bandwidth;
    bool run_cache;
    bool run_pcie;
    bool run_nccl;
    bool run_scheduling;
    bool run_fragmentation;
    bool run_error;
    bool run_all_extended;
    bool worker_mode;
    char worker_test_id[64];
    char worker_args[256];
} cli_options_t;

static void print_usage(const char *prog_name) {
    printf("GPU Virtualization Benchmark Tool v1.0.0\n\n");
    printf("Usage: %s [options]\n\n", prog_name);
    printf("Options:\n");
    printf("  --system <name>       System name label (default: native)\n");
    printf("  --config <file>       Load configuration from file\n");
    printf("  --output <dir>        Output directory (default: ./benchmarks)\n");
    printf("  --iterations <n>      Benchmark iterations (default: 100)\n");
    printf("  --warmup <n>          Warmup iterations (default: 10)\n");
    printf("  --overhead            Run overhead benchmarks only\n");
    printf("  --isolation           Run isolation benchmarks only\n");
    printf("  --llm                 Run LLM benchmarks only\n");
    printf("  --bandwidth           Run memory bandwidth isolation benchmarks\n");
    printf("  --cache               Run cache isolation benchmarks\n");
    printf("  --pcie                Run PCIe bandwidth benchmarks\n");
    printf("  --nccl                Run NCCL/P2P communication benchmarks\n");
    printf("  --scheduling          Run scheduling/context switch benchmarks\n");
    printf("  --fragmentation       Run memory fragmentation benchmarks\n");
    printf("  --error               Run error recovery benchmarks\n");
    printf("  --all-extended        Run all extended benchmarks\n");
    printf("  --verbose             Verbose output\n");
    printf("  --help                Show this help\n");
}

static int parse_cli(int argc, char *argv[], cli_options_t *opts) {
    /* Defaults */
    memset(opts, 0, sizeof(*opts));
    strcpy(opts->system_name, "native");
    strcpy(opts->output_dir, "./benchmarks");
    opts->iterations = 100;
    opts->warmup = 10;
    opts->run_overhead = true;
    opts->run_isolation = true;
    opts->run_llm = true;
    opts->run_bandwidth = false;
    opts->run_cache = false;
    opts->run_pcie = false;
    opts->run_nccl = false;
    opts->run_scheduling = false;
    opts->run_fragmentation = false;
    opts->run_error = false;
    opts->run_all_extended = false;
    opts->worker_mode = false;

    /* Check for --worker mode first (used by subprocess for isolation tests) */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--worker") == 0 && i + 1 < argc) {
            opts->worker_mode = true;
            strncpy(opts->worker_test_id, argv[i + 1], sizeof(opts->worker_test_id) - 1);
            /* Check for --worker-args */
            if (i + 3 < argc && strcmp(argv[i + 2], "--worker-args") == 0) {
                strncpy(opts->worker_args, argv[i + 3], sizeof(opts->worker_args) - 1);
            }
            return 0;
        }
    }

    static struct option long_options[] = {
        {"system",        required_argument, 0, 's'},
        {"config",        required_argument, 0, 'c'},
        {"output",        required_argument, 0, 'o'},
        {"iterations",    required_argument, 0, 'i'},
        {"warmup",        required_argument, 0, 'w'},
        {"overhead",      no_argument,       0, 'O'},
        {"isolation",     no_argument,       0, 'I'},
        {"llm",           no_argument,       0, 'L'},
        {"bandwidth",     no_argument,       0, 'B'},
        {"cache",         no_argument,       0, 'C'},
        {"pcie",          no_argument,       0, 'P'},
        {"nccl",          no_argument,       0, 'N'},
        {"scheduling",    no_argument,       0, 'S'},
        {"fragmentation", no_argument,       0, 'F'},
        {"error",         no_argument,       0, 'E'},
        {"all-extended",  no_argument,       0, 'A'},
        {"verbose",       no_argument,       0, 'v'},
        {"help",          no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    bool category_specified = false;

    while ((opt = getopt_long(argc, argv, "s:c:o:i:w:OILBCPNSFEAvh", long_options, NULL)) != -1) {
        switch (opt) {
            case 's':
                strncpy(opts->system_name, optarg, sizeof(opts->system_name) - 1);
                break;
            case 'c':
                strncpy(opts->config_file, optarg, sizeof(opts->config_file) - 1);
                break;
            case 'o':
                strncpy(opts->output_dir, optarg, sizeof(opts->output_dir) - 1);
                break;
            case 'i':
                opts->iterations = atoi(optarg);
                break;
            case 'w':
                opts->warmup = atoi(optarg);
                break;
            case 'O':
                if (!category_specified) {
                    opts->run_overhead = opts->run_isolation = opts->run_llm = false;
                    category_specified = true;
                }
                opts->run_overhead = true;
                break;
            case 'I':
                if (!category_specified) {
                    opts->run_overhead = opts->run_isolation = opts->run_llm = false;
                    category_specified = true;
                }
                opts->run_isolation = true;
                break;
            case 'L':
                if (!category_specified) {
                    opts->run_overhead = opts->run_isolation = opts->run_llm = false;
                    category_specified = true;
                }
                opts->run_llm = true;
                break;
            case 'B':
                if (!category_specified) {
                    opts->run_overhead = opts->run_isolation = opts->run_llm = false;
                    category_specified = true;
                }
                opts->run_bandwidth = true;
                break;
            case 'C':
                if (!category_specified) {
                    opts->run_overhead = opts->run_isolation = opts->run_llm = false;
                    category_specified = true;
                }
                opts->run_cache = true;
                break;
            case 'P':
                if (!category_specified) {
                    opts->run_overhead = opts->run_isolation = opts->run_llm = false;
                    category_specified = true;
                }
                opts->run_pcie = true;
                break;
            case 'N':
                if (!category_specified) {
                    opts->run_overhead = opts->run_isolation = opts->run_llm = false;
                    category_specified = true;
                }
                opts->run_nccl = true;
                break;
            case 'S':
                if (!category_specified) {
                    opts->run_overhead = opts->run_isolation = opts->run_llm = false;
                    category_specified = true;
                }
                opts->run_scheduling = true;
                break;
            case 'F':
                if (!category_specified) {
                    opts->run_overhead = opts->run_isolation = opts->run_llm = false;
                    category_specified = true;
                }
                opts->run_fragmentation = true;
                break;
            case 'E':
                if (!category_specified) {
                    opts->run_overhead = opts->run_isolation = opts->run_llm = false;
                    category_specified = true;
                }
                opts->run_error = true;
                break;
            case 'A':
                /* Run all extended benchmarks */
                opts->run_bandwidth = true;
                opts->run_cache = true;
                opts->run_pcie = true;
                opts->run_nccl = true;
                opts->run_scheduling = true;
                opts->run_fragmentation = true;
                opts->run_error = true;
                opts->run_all_extended = true;
                break;
            case 'v':
                opts->verbose = true;
                break;
            case 'h':
                print_usage(argv[0]);
                exit(0);
            default:
                return -1;
        }
    }
    return 0;
}

/* Get current time in microseconds */
static double get_time_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1e6 + (double)tv.tv_usec;
}

/* Get unit string for metric */
static const char* get_unit_string(metric_unit_t unit) {
    switch (unit) {
        case METRIC_UNIT_NANOSECONDS: return "ns";
        case METRIC_UNIT_MICROSECONDS: return "us";
        case METRIC_UNIT_MILLISECONDS: return "ms";
        case METRIC_UNIT_SECONDS: return "s";
        case METRIC_UNIT_PERCENTAGE: return "%";
        case METRIC_UNIT_BOOLEAN: return "bool";
        case METRIC_UNIT_RATIO: return "ratio";
        case METRIC_UNIT_THROUGHPUT: return "ops/s";
        case METRIC_UNIT_BANDWIDTH: return "GB/s";
        case METRIC_UNIT_FLOPS: return "TFLOPS";
        case METRIC_UNIT_TOKENS_PER_SEC: return "tok/s";
        case METRIC_UNIT_ALLOCS_PER_SEC: return "alloc/s";
        default: return "?";
    }
}

/* Save results to JSON */
static int save_json(const char *filepath, const char *system_name,
                     metric_result_t *results, int count, double duration) {
    FILE *f = fopen(filepath, "w");
    if (!f) {
        perror("Failed to open output file");
        return -1;
    }

    /* Get system info */
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    time_t now = time(NULL);
    char timestamp[64];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%SZ", gmtime(&now));

    fprintf(f, "{\n");
    fprintf(f, "  \"benchmark_version\": \"1.0.0\",\n");
    fprintf(f, "  \"timestamp\": \"%s\",\n", timestamp);
    fprintf(f, "  \"system_name\": \"%s\",\n", system_name);
    fprintf(f, "  \"duration_seconds\": %.3f,\n", duration);
    fprintf(f, "  \"gpu\": {\n");
    fprintf(f, "    \"name\": \"%s\",\n", prop.name);
    fprintf(f, "    \"compute_capability\": \"%d.%d\",\n", prop.major, prop.minor);
    fprintf(f, "    \"total_memory_mb\": %zu,\n", total_mem / (1024 * 1024));
    fprintf(f, "    \"sm_count\": %d\n", prop.multiProcessorCount);
    fprintf(f, "  },\n");
    fprintf(f, "  \"hami_detected\": %s,\n",
            getenv("CUDA_DEVICE_MEMORY_LIMIT") ? "true" : "false");

    fprintf(f, "  \"metrics\": [\n");
    for (int i = 0; i < count; i++) {
        metric_result_t *m = &results[i];
        fprintf(f, "    {\n");
        fprintf(f, "      \"id\": \"%s\",\n", m->metric_id);
        fprintf(f, "      \"name\": \"%s\",\n", m->name);
        fprintf(f, "      \"unit\": \"%s\",\n", m->unit);
        fprintf(f, "      \"success\": %s,\n", m->valid ? "true" : "false");
        if (m->valid) {
            fprintf(f, "      \"value\": %.6f,\n", m->value);
            fprintf(f, "      \"stats\": {\n");
            fprintf(f, "        \"mean\": %.6f,\n", m->stats.mean);
            fprintf(f, "        \"std_dev\": %.6f,\n", m->stats.std_dev);
            fprintf(f, "        \"min\": %.6f,\n", m->stats.min);
            fprintf(f, "        \"max\": %.6f,\n", m->stats.max);
            fprintf(f, "        \"p50\": %.6f,\n", m->stats.p50);
            fprintf(f, "        \"p90\": %.6f,\n", m->stats.p90);
            fprintf(f, "        \"p95\": %.6f,\n", m->stats.p95);
            fprintf(f, "        \"p99\": %.6f,\n", m->stats.p99);
            fprintf(f, "        \"count\": %lu\n", (unsigned long)m->stats.count);
            fprintf(f, "      }\n");
        } else {
            fprintf(f, "      \"error\": \"%s\"\n", m->error_msg);
        }
        fprintf(f, "    }%s\n", (i < count - 1) ? "," : "");
    }
    fprintf(f, "  ]\n");
    fprintf(f, "}\n");

    fclose(f);
    return 0;
}

/* Main */
int main(int argc, char *argv[]) {
    cli_options_t opts;

    if (parse_cli(argc, argv, &opts) != 0) {
        print_usage(argv[0]);
        return 1;
    }

    /* Worker mode - used by isolation tests for subprocess execution */
    if (opts.worker_mode) {
        return dispatch_worker(opts.worker_test_id, opts.worker_args[0] ? opts.worker_args : NULL);
    }

    /* Setup signal handler */
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    /* Check CUDA */
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        fprintf(stderr, "ERROR: No CUDA devices found\n");
        return 1;
    }

    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    /* Check for HAMI */
    const char *mem_limit = getenv("CUDA_DEVICE_MEMORY_LIMIT");
    const char *sm_limit = getenv("CUDA_DEVICE_SM_LIMIT");
    bool hami_detected = (mem_limit != NULL || sm_limit != NULL);

    printf("\n");
    printf("================================================================================\n");
    printf("  GPU VIRTUALIZATION BENCHMARK\n");
    printf("================================================================================\n");
    printf("  System: %s\n", opts.system_name);
    printf("  GPU: %s (SM: %d, Mem: %zu MB)\n",
           prop.name, prop.multiProcessorCount, total_mem / (1024*1024));
    printf("  HAMI: %s\n", hami_detected ? "Detected" : "Not detected");
    if (hami_detected) {
        if (mem_limit) printf("    Memory Limit: %s\n", mem_limit);
        if (sm_limit) printf("    SM Limit: %s%%\n", sm_limit);
    }
    printf("  Iterations: %d (warmup: %d)\n", opts.iterations, opts.warmup);
    printf("================================================================================\n\n");

    /* Create output directory */
    mkdir(opts.output_dir, 0755);

    /* Prepare results storage */
    metric_result_t *all_results = (metric_result_t*)calloc(TOTAL_METRIC_COUNT, sizeof(metric_result_t));
    int result_count = 0;

    double start_time = get_time_us();

    /* Create benchmark config */
    benchmark_config_t config;
    memset(&config, 0, sizeof(config));
    config.options.iterations = opts.iterations;
    config.options.warmup_iterations = opts.warmup;
    config.options.verbose = opts.verbose;

    /* Run overhead benchmarks */
    if (opts.run_overhead && !g_interrupted) {
        printf("Running OVERHEAD benchmarks...\n");
        benchmark_result_t oh_results;
        memset(&oh_results, 0, sizeof(oh_results));

        if (bench_run_overhead(&config, &oh_results) == 0) {
            /* Copy results */
            for (int i = 0; i < oh_results.result_count && result_count < TOTAL_METRIC_COUNT; i++) {
                memcpy(&all_results[result_count], &oh_results.results[i], sizeof(metric_result_t));
                /* Set name and unit from definitions */
                for (int j = 0; j < TOTAL_METRIC_COUNT; j++) {
                    if (strcmp(METRIC_DEFINITIONS[j].id, oh_results.results[i].metric_id) == 0) {
                        strncpy(all_results[result_count].name, METRIC_DEFINITIONS[j].name,
                                sizeof(all_results[result_count].name) - 1);
                        strncpy(all_results[result_count].unit,
                                get_unit_string(METRIC_DEFINITIONS[j].unit),
                                sizeof(all_results[result_count].unit) - 1);
                        break;
                    }
                }
                result_count++;
            }
            if (oh_results.results) free(oh_results.results);
        }
        printf("\n");
    }

    /* Run isolation benchmarks */
    if (opts.run_isolation && !g_interrupted) {
        printf("Running ISOLATION benchmarks...\n");
        benchmark_result_t is_results;
        memset(&is_results, 0, sizeof(is_results));

        if (bench_run_isolation(&config, &is_results) == 0) {
            for (int i = 0; i < is_results.result_count && result_count < TOTAL_METRIC_COUNT; i++) {
                memcpy(&all_results[result_count], &is_results.results[i], sizeof(metric_result_t));
                for (int j = 0; j < TOTAL_METRIC_COUNT; j++) {
                    if (strcmp(METRIC_DEFINITIONS[j].id, is_results.results[i].metric_id) == 0) {
                        strncpy(all_results[result_count].name, METRIC_DEFINITIONS[j].name,
                                sizeof(all_results[result_count].name) - 1);
                        strncpy(all_results[result_count].unit,
                                get_unit_string(METRIC_DEFINITIONS[j].unit),
                                sizeof(all_results[result_count].unit) - 1);
                        break;
                    }
                }
                result_count++;
            }
            if (is_results.results) free(is_results.results);
        }
        printf("\n");
    }

    /* Run LLM benchmarks */
    if (opts.run_llm && !g_interrupted) {
        printf("Running LLM benchmarks...\n");
        benchmark_result_t llm_results;
        memset(&llm_results, 0, sizeof(llm_results));

        if (bench_run_llm(&config, &llm_results) == 0) {
            for (int i = 0; i < llm_results.result_count && result_count < TOTAL_METRIC_COUNT; i++) {
                memcpy(&all_results[result_count], &llm_results.results[i], sizeof(metric_result_t));
                for (int j = 0; j < TOTAL_METRIC_COUNT; j++) {
                    if (strcmp(METRIC_DEFINITIONS[j].id, llm_results.results[i].metric_id) == 0) {
                        strncpy(all_results[result_count].name, METRIC_DEFINITIONS[j].name,
                                sizeof(all_results[result_count].name) - 1);
                        strncpy(all_results[result_count].unit,
                                get_unit_string(METRIC_DEFINITIONS[j].unit),
                                sizeof(all_results[result_count].unit) - 1);
                        break;
                    }
                }
                result_count++;
            }
            if (llm_results.results) free(llm_results.results);
        }
        printf("\n");
    }

    /* ========== Extended Benchmark Categories ========== */

    /* Helper function to convert bench_result_t to metric_result_t */
    #define CONVERT_EXTENDED_RESULTS(results_arr, count_var) \
        for (int i = 0; i < count_var && result_count < TOTAL_METRIC_COUNT; i++) { \
            strncpy(all_results[result_count].metric_id, results_arr[i].metric_id, \
                    sizeof(all_results[result_count].metric_id) - 1); \
            strncpy(all_results[result_count].name, results_arr[i].name, \
                    sizeof(all_results[result_count].name) - 1); \
            strncpy(all_results[result_count].unit, results_arr[i].unit, \
                    sizeof(all_results[result_count].unit) - 1); \
            all_results[result_count].value = results_arr[i].value; \
            all_results[result_count].stats.mean = results_arr[i].value; \
            all_results[result_count].stats.std_dev = results_arr[i].stddev; \
            all_results[result_count].stats.count = results_arr[i].iterations; \
            all_results[result_count].valid = results_arr[i].success; \
            strncpy(all_results[result_count].error_msg, results_arr[i].error_message, \
                    sizeof(all_results[result_count].error_msg) - 1); \
            result_count++; \
        }

    /* Run bandwidth benchmarks */
    if (opts.run_bandwidth && !g_interrupted) {
        printf("Running BANDWIDTH benchmarks...\n");
        bench_config_t bw_config = {opts.iterations, opts.warmup, opts.verbose};
        bench_result_t bw_results[10];
        int bw_count = 0;
        bench_run_bandwidth(&bw_config, bw_results, &bw_count);
        CONVERT_EXTENDED_RESULTS(bw_results, bw_count);
        printf("\n");
    }

    /* Run cache benchmarks */
    if (opts.run_cache && !g_interrupted) {
        printf("Running CACHE benchmarks...\n");
        bench_config_t cache_config = {opts.iterations, opts.warmup, opts.verbose};
        bench_result_t cache_results[10];
        int cache_count = 0;
        bench_run_cache(&cache_config, cache_results, &cache_count);
        CONVERT_EXTENDED_RESULTS(cache_results, cache_count);
        printf("\n");
    }

    /* Run PCIe benchmarks */
    if (opts.run_pcie && !g_interrupted) {
        printf("Running PCIE benchmarks...\n");
        bench_config_t pcie_config = {opts.iterations, opts.warmup, opts.verbose};
        bench_result_t pcie_results[10];
        int pcie_count = 0;
        bench_run_pcie(&pcie_config, pcie_results, &pcie_count);
        CONVERT_EXTENDED_RESULTS(pcie_results, pcie_count);
        printf("\n");
    }

    /* Run NCCL benchmarks */
    if (opts.run_nccl && !g_interrupted) {
        printf("Running NCCL benchmarks...\n");
        bench_config_t nccl_config = {opts.iterations, opts.warmup, opts.verbose};
        bench_result_t nccl_results[10];
        int nccl_count = 0;
        bench_run_nccl(&nccl_config, nccl_results, &nccl_count);
        CONVERT_EXTENDED_RESULTS(nccl_results, nccl_count);
        printf("\n");
    }

    /* Run scheduling benchmarks */
    if (opts.run_scheduling && !g_interrupted) {
        printf("Running SCHEDULING benchmarks...\n");
        bench_config_t sched_config = {opts.iterations, opts.warmup, opts.verbose};
        bench_result_t sched_results[10];
        int sched_count = 0;
        bench_run_scheduling(&sched_config, sched_results, &sched_count);
        CONVERT_EXTENDED_RESULTS(sched_results, sched_count);
        printf("\n");
    }

    /* Run fragmentation benchmarks */
    if (opts.run_fragmentation && !g_interrupted) {
        printf("Running FRAGMENTATION benchmarks...\n");
        bench_config_t frag_config = {opts.iterations, opts.warmup, opts.verbose};
        bench_result_t frag_results[10];
        int frag_count = 0;
        bench_run_fragmentation(&frag_config, frag_results, &frag_count);
        CONVERT_EXTENDED_RESULTS(frag_results, frag_count);
        printf("\n");
    }

    /* Run error recovery benchmarks */
    if (opts.run_error && !g_interrupted) {
        printf("Running ERROR RECOVERY benchmarks...\n");
        bench_config_t error_config = {opts.iterations, opts.warmup, opts.verbose};
        bench_result_t error_results[10];
        int error_count = 0;
        bench_run_error(&error_config, error_results, &error_count);
        CONVERT_EXTENDED_RESULTS(error_results, error_count);
        printf("\n");
    }

    double total_duration = (get_time_us() - start_time) / 1e6;

    /* Print summary */
    printf("================================================================================\n");
    printf("  RESULTS SUMMARY\n");
    printf("================================================================================\n\n");

    int success_count = 0;
    for (int i = 0; i < result_count; i++) {
        metric_result_t *m = &all_results[i];
        if (m->valid) {
            printf("  %-8s %-35s: %10.3f %-8s (std: %.3f)\n",
                   m->metric_id, m->name, m->stats.mean, m->unit, m->stats.std_dev);
            success_count++;
        } else {
            printf("  %-8s %-35s: FAILED - %s\n",
                   m->metric_id, m->name, m->error_msg);
        }
    }

    printf("\n");
    printf("  Completed: %d/%d metrics\n", success_count, result_count);
    printf("  Duration: %.2f seconds\n", total_duration);
    printf("\n");

    /* Save JSON */
    char json_path[512];
    snprintf(json_path, sizeof(json_path), "%s/%s_results.json", opts.output_dir, opts.system_name);
    if (save_json(json_path, opts.system_name, all_results, result_count, total_duration) == 0) {
        printf("  Results saved to: %s\n", json_path);
    }

    printf("\n================================================================================\n");
    printf("  BENCHMARK COMPLETE\n");
    printf("================================================================================\n\n");

    free(all_results);
    return 0;
}
