/*
 * GPU Virtualization Performance Evaluation Tool
 * Results collection, storage, and reporting
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <dirent.h>

#include "include/benchmark.h"
#include "include/metrics.h"

/*
 * ============================================================================
 * Directory and File Management
 * ============================================================================
 */

static int ensure_directory_exists(const char *path) {
    struct stat st = {0};
    if (stat(path, &st) == -1) {
        if (mkdir(path, 0755) != 0) {
            fprintf(stderr, "ERROR: Failed to create directory %s: %s\n",
                    path, strerror(errno));
            return -1;
        }
    }
    return 0;
}

static void get_timestamp_string(char *buffer, size_t size) {
    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);
    strftime(buffer, size, "%Y%m%d_%H%M%S", tm_info);
}

static void get_date_string(char *buffer, size_t size) {
    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);
    strftime(buffer, size, "%Y-%m-%d %H:%M:%S", tm_info);
}

/*
 * ============================================================================
 * JSON Report Generation
 * ============================================================================
 */

static void escape_json_string(const char *input, char *output, size_t output_size) {
    size_t j = 0;
    for (size_t i = 0; input[i] && j < output_size - 2; i++) {
        switch (input[i]) {
            case '"':  output[j++] = '\\'; output[j++] = '"'; break;
            case '\\': output[j++] = '\\'; output[j++] = '\\'; break;
            case '\n': output[j++] = '\\'; output[j++] = 'n'; break;
            case '\r': output[j++] = '\\'; output[j++] = 'r'; break;
            case '\t': output[j++] = '\\'; output[j++] = 't'; break;
            default:   output[j++] = input[i]; break;
        }
    }
    output[j] = '\0';
}

int report_save_json(const benchmark_result_t *result, const char *output_dir) {
    char dir_path[512];
    char file_path[512];
    char timestamp[64];

    /* Create output directory structure */
    snprintf(dir_path, sizeof(dir_path), "%s", output_dir);
    if (ensure_directory_exists(dir_path) != 0) return -1;

    snprintf(dir_path, sizeof(dir_path), "%s/%s", output_dir, result->system_name);
    if (ensure_directory_exists(dir_path) != 0) return -1;

    /* Generate filename */
    get_timestamp_string(timestamp, sizeof(timestamp));
    snprintf(file_path, sizeof(file_path), "%s/%s_%s.json",
             dir_path, result->system_name, timestamp);

    FILE *f = fopen(file_path, "w");
    if (f == NULL) {
        fprintf(stderr, "ERROR: Failed to create file %s: %s\n",
                file_path, strerror(errno));
        return -1;
    }

    char date_str[64];
    get_date_string(date_str, sizeof(date_str));

    fprintf(f, "{\n");
    fprintf(f, "  \"benchmark_version\": \"1.0.0\",\n");
    fprintf(f, "  \"timestamp\": \"%s\",\n", date_str);
    fprintf(f, "  \"system\": {\n");
    fprintf(f, "    \"name\": \"%s\",\n", result->system_name);
    fprintf(f, "    \"version\": \"%s\"\n", result->system_version);
    fprintf(f, "  },\n");

    fprintf(f, "  \"configuration\": {\n");
    fprintf(f, "    \"warmup_iterations\": %d,\n", result->config.warmup_iterations);
    fprintf(f, "    \"benchmark_iterations\": %d,\n", result->config.benchmark_iterations);
    fprintf(f, "    \"num_processes\": %d,\n", result->config.num_processes);
    fprintf(f, "    \"memory_limit_mb\": %lu,\n", result->config.memory_limit_mb);
    fprintf(f, "    \"compute_limit_percent\": %d\n", result->config.compute_limit_percent);
    fprintf(f, "  },\n");

    fprintf(f, "  \"total_duration_seconds\": %.3f,\n", result->total_duration_seconds);

    /* Metrics array */
    fprintf(f, "  \"metrics\": [\n");
    for (int i = 0; i < result->num_metrics; i++) {
        const metric_result_t *m = &result->metrics[i];
        char escaped_name[256];
        escape_json_string(m->name, escaped_name, sizeof(escaped_name));

        fprintf(f, "    {\n");
        fprintf(f, "      \"id\": \"%s\",\n", m->metric_id);
        fprintf(f, "      \"name\": \"%s\",\n", escaped_name);
        fprintf(f, "      \"unit\": \"%s\",\n", m->unit);
        fprintf(f, "      \"success\": %s,\n", m->success ? "true" : "false");

        if (m->success) {
            fprintf(f, "      \"statistics\": {\n");
            fprintf(f, "        \"count\": %lu,\n", m->stats.count);
            fprintf(f, "        \"mean\": %.6f,\n", m->stats.mean);
            fprintf(f, "        \"median\": %.6f,\n", m->stats.median);
            fprintf(f, "        \"std_dev\": %.6f,\n", m->stats.std_dev);
            fprintf(f, "        \"min\": %.6f,\n", m->stats.min);
            fprintf(f, "        \"max\": %.6f,\n", m->stats.max);
            fprintf(f, "        \"p50\": %.6f,\n", m->stats.p50);
            fprintf(f, "        \"p90\": %.6f,\n", m->stats.p90);
            fprintf(f, "        \"p95\": %.6f,\n", m->stats.p95);
            fprintf(f, "        \"p99\": %.6f,\n", m->stats.p99);
            fprintf(f, "        \"p999\": %.6f\n", m->stats.p999);
            fprintf(f, "      },\n");

            fprintf(f, "      \"mig_comparison\": {\n");
            fprintf(f, "        \"mig_expected\": %.6f,\n", m->mig_expected);
            fprintf(f, "        \"mig_gap_percent\": %.2f,\n", m->mig_gap_percent);
            fprintf(f, "        \"normalized_score\": %.4f\n", m->normalized_score);
            fprintf(f, "      }\n");
        } else {
            char escaped_error[1024];
            escape_json_string(m->error_message, escaped_error, sizeof(escaped_error));
            fprintf(f, "      \"error\": \"%s\"\n", escaped_error);
        }

        fprintf(f, "    }%s\n", (i < result->num_metrics - 1) ? "," : "");
    }
    fprintf(f, "  ],\n");

    /* Summary scores */
    fprintf(f, "  \"summary\": {\n");
    fprintf(f, "    \"overhead_score\": %.4f,\n", result->overhead_score);
    fprintf(f, "    \"isolation_score\": %.4f,\n", result->isolation_score);
    fprintf(f, "    \"llm_score\": %.4f,\n", result->llm_score);
    fprintf(f, "    \"overall_score\": %.4f,\n", result->overall_score);
    fprintf(f, "    \"mig_parity_percent\": %.2f\n", result->mig_parity_percent);
    fprintf(f, "  }\n");

    fprintf(f, "}\n");

    fclose(f);
    printf("JSON report saved: %s\n", file_path);
    return 0;
}

/*
 * ============================================================================
 * CSV Report Generation
 * ============================================================================
 */

int report_save_csv(const benchmark_result_t *result, const char *output_dir) {
    char dir_path[512];
    char file_path[512];
    char timestamp[64];

    snprintf(dir_path, sizeof(dir_path), "%s/%s", output_dir, result->system_name);
    if (ensure_directory_exists(output_dir) != 0) return -1;
    if (ensure_directory_exists(dir_path) != 0) return -1;

    get_timestamp_string(timestamp, sizeof(timestamp));
    snprintf(file_path, sizeof(file_path), "%s/%s_%s.csv",
             dir_path, result->system_name, timestamp);

    FILE *f = fopen(file_path, "w");
    if (f == NULL) {
        fprintf(stderr, "ERROR: Failed to create file %s\n", file_path);
        return -1;
    }

    /* Header */
    fprintf(f, "metric_id,name,unit,success,count,mean,median,std_dev,min,max,"
               "p50,p90,p95,p99,p999,mig_expected,mig_gap_percent,normalized_score\n");

    /* Data rows */
    for (int i = 0; i < result->num_metrics; i++) {
        const metric_result_t *m = &result->metrics[i];

        fprintf(f, "%s,\"%s\",%s,%d,", m->metric_id, m->name, m->unit, m->success ? 1 : 0);

        if (m->success) {
            fprintf(f, "%lu,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
                       "%.6f,%.2f,%.4f\n",
                    m->stats.count, m->stats.mean, m->stats.median, m->stats.std_dev,
                    m->stats.min, m->stats.max, m->stats.p50, m->stats.p90,
                    m->stats.p95, m->stats.p99, m->stats.p999,
                    m->mig_expected, m->mig_gap_percent, m->normalized_score);
        } else {
            fprintf(f, "0,0,0,0,0,0,0,0,0,0,0,0,0,0\n");
        }
    }

    fclose(f);
    printf("CSV report saved: %s\n", file_path);
    return 0;
}

/*
 * ============================================================================
 * Human-Readable Text Report
 * ============================================================================
 */

static const char *get_score_grade(double score) {
    if (score >= 0.95) return "A+ (Excellent)";
    if (score >= 0.90) return "A  (Very Good)";
    if (score >= 0.85) return "B+ (Good)";
    if (score >= 0.80) return "B  (Acceptable)";
    if (score >= 0.70) return "C  (Fair)";
    if (score >= 0.60) return "D  (Poor)";
    return "F  (Failing)";
}

static const char *get_gap_indicator(double gap_percent) {
    if (gap_percent <= 5.0) return "[==]";   /* Excellent - within 5% */
    if (gap_percent <= 15.0) return "[= ]";  /* Good - within 15% */
    if (gap_percent <= 30.0) return "[- ]";  /* Fair - within 30% */
    if (gap_percent <= 50.0) return "[--]";  /* Poor - within 50% */
    return "[!!]";                            /* Critical - over 50% */
}

int report_save_text(const benchmark_result_t *result, const char *output_dir) {
    char dir_path[512];
    char file_path[512];
    char timestamp[64];

    snprintf(dir_path, sizeof(dir_path), "%s/%s", output_dir, result->system_name);
    if (ensure_directory_exists(output_dir) != 0) return -1;
    if (ensure_directory_exists(dir_path) != 0) return -1;

    get_timestamp_string(timestamp, sizeof(timestamp));
    snprintf(file_path, sizeof(file_path), "%s/%s_%s.txt",
             dir_path, result->system_name, timestamp);

    FILE *f = fopen(file_path, "w");
    if (f == NULL) {
        fprintf(stderr, "ERROR: Failed to create file %s\n", file_path);
        return -1;
    }

    char date_str[64];
    get_date_string(date_str, sizeof(date_str));

    fprintf(f, "================================================================================\n");
    fprintf(f, "       GPU VIRTUALIZATION PERFORMANCE BENCHMARK REPORT\n");
    fprintf(f, "================================================================================\n\n");

    fprintf(f, "System:          %s v%s\n", result->system_name, result->system_version);
    fprintf(f, "Timestamp:       %s\n", date_str);
    fprintf(f, "Duration:        %.2f seconds\n\n", result->total_duration_seconds);

    fprintf(f, "Configuration:\n");
    fprintf(f, "  - Warmup iterations:    %d\n", result->config.warmup_iterations);
    fprintf(f, "  - Benchmark iterations: %d\n", result->config.benchmark_iterations);
    fprintf(f, "  - Processes:            %d\n", result->config.num_processes);
    fprintf(f, "  - Memory limit:         %lu MB\n", result->config.memory_limit_mb);
    fprintf(f, "  - Compute limit:        %d%%\n\n", result->config.compute_limit_percent);

    /* Summary scores */
    fprintf(f, "================================================================================\n");
    fprintf(f, "                           SUMMARY SCORES\n");
    fprintf(f, "================================================================================\n\n");

    fprintf(f, "  Overall Score:    %.1f%% - %s\n",
            result->overall_score * 100.0, get_score_grade(result->overall_score));
    fprintf(f, "  MIG Parity:       %.1f%%\n\n", result->mig_parity_percent);

    fprintf(f, "  Category Breakdown:\n");
    fprintf(f, "  +-----------------+--------+-----------------+\n");
    fprintf(f, "  | Category        | Score  | Grade           |\n");
    fprintf(f, "  +-----------------+--------+-----------------+\n");
    fprintf(f, "  | Overhead        | %5.1f%% | %-15s |\n",
            result->overhead_score * 100.0, get_score_grade(result->overhead_score));
    fprintf(f, "  | Isolation       | %5.1f%% | %-15s |\n",
            result->isolation_score * 100.0, get_score_grade(result->isolation_score));
    fprintf(f, "  | LLM Performance | %5.1f%% | %-15s |\n",
            result->llm_score * 100.0, get_score_grade(result->llm_score));
    fprintf(f, "  +-----------------+--------+-----------------+\n\n");

    /* Overhead Metrics */
    fprintf(f, "================================================================================\n");
    fprintf(f, "                        OVERHEAD METRICS (OH-001 to OH-010)\n");
    fprintf(f, "================================================================================\n\n");

    fprintf(f, "  %-8s %-35s %12s %12s %8s %s\n",
            "ID", "Metric", "Measured", "MIG Exp.", "Gap%", "Status");
    fprintf(f, "  %-8s %-35s %12s %12s %8s %s\n",
            "--------", "-----------------------------------", "------------",
            "------------", "--------", "------");

    for (int i = 0; i < result->num_metrics; i++) {
        const metric_result_t *m = &result->metrics[i];
        if (strncmp(m->metric_id, "OH-", 3) != 0) continue;

        if (m->success) {
            fprintf(f, "  %-8s %-35.35s %12.3f %12.3f %7.1f%% %s\n",
                    m->metric_id, m->name,
                    m->stats.mean, m->mig_expected,
                    m->mig_gap_percent, get_gap_indicator(m->mig_gap_percent));
        } else {
            fprintf(f, "  %-8s %-35.35s %12s %12s %8s %s\n",
                    m->metric_id, m->name, "FAILED", "-", "-", "[!!]");
        }
    }
    fprintf(f, "\n");

    /* Isolation Metrics */
    fprintf(f, "================================================================================\n");
    fprintf(f, "                       ISOLATION METRICS (IS-001 to IS-010)\n");
    fprintf(f, "================================================================================\n\n");

    fprintf(f, "  %-8s %-35s %12s %12s %8s %s\n",
            "ID", "Metric", "Measured", "MIG Exp.", "Gap%", "Status");
    fprintf(f, "  %-8s %-35s %12s %12s %8s %s\n",
            "--------", "-----------------------------------", "------------",
            "------------", "--------", "------");

    for (int i = 0; i < result->num_metrics; i++) {
        const metric_result_t *m = &result->metrics[i];
        if (strncmp(m->metric_id, "IS-", 3) != 0) continue;

        if (m->success) {
            fprintf(f, "  %-8s %-35.35s %12.3f %12.3f %7.1f%% %s\n",
                    m->metric_id, m->name,
                    m->stats.mean, m->mig_expected,
                    m->mig_gap_percent, get_gap_indicator(m->mig_gap_percent));
        } else {
            fprintf(f, "  %-8s %-35.35s %12s %12s %8s %s\n",
                    m->metric_id, m->name, "FAILED", "-", "-", "[!!]");
        }
    }
    fprintf(f, "\n");

    /* LLM Metrics */
    fprintf(f, "================================================================================\n");
    fprintf(f, "                       LLM METRICS (LLM-001 to LLM-010)\n");
    fprintf(f, "================================================================================\n\n");

    fprintf(f, "  %-8s %-35s %12s %12s %8s %s\n",
            "ID", "Metric", "Measured", "MIG Exp.", "Gap%", "Status");
    fprintf(f, "  %-8s %-35s %12s %12s %8s %s\n",
            "--------", "-----------------------------------", "------------",
            "------------", "--------", "------");

    for (int i = 0; i < result->num_metrics; i++) {
        const metric_result_t *m = &result->metrics[i];
        if (strncmp(m->metric_id, "LLM-", 4) != 0) continue;

        if (m->success) {
            fprintf(f, "  %-8s %-35.35s %12.3f %12.3f %7.1f%% %s\n",
                    m->metric_id, m->name,
                    m->stats.mean, m->mig_expected,
                    m->mig_gap_percent, get_gap_indicator(m->mig_gap_percent));
        } else {
            fprintf(f, "  %-8s %-35.35s %12s %12s %8s %s\n",
                    m->metric_id, m->name, "FAILED", "-", "-", "[!!]");
        }
    }
    fprintf(f, "\n");

    /* Legend */
    fprintf(f, "================================================================================\n");
    fprintf(f, "Legend: [==] Excellent (<5%% gap)  [= ] Good (<15%%)  [- ] Fair (<30%%)\n");
    fprintf(f, "        [--] Poor (<50%%)         [!!] Critical (>50%% gap or failed)\n");
    fprintf(f, "================================================================================\n");

    fclose(f);
    printf("Text report saved: %s\n", file_path);
    return 0;
}

/*
 * ============================================================================
 * Comparison Report - Compare Two Systems
 * ============================================================================
 */

int report_save_comparison(const benchmark_result_t *baseline,
                           const benchmark_result_t *improved,
                           const char *output_dir) {
    char file_path[512];
    char timestamp[64];

    if (ensure_directory_exists(output_dir) != 0) return -1;

    get_timestamp_string(timestamp, sizeof(timestamp));
    snprintf(file_path, sizeof(file_path), "%s/comparison_%s_vs_%s_%s.txt",
             output_dir, baseline->system_name, improved->system_name, timestamp);

    FILE *f = fopen(file_path, "w");
    if (f == NULL) {
        fprintf(stderr, "ERROR: Failed to create comparison report\n");
        return -1;
    }

    char date_str[64];
    get_date_string(date_str, sizeof(date_str));

    fprintf(f, "================================================================================\n");
    fprintf(f, "       GPU VIRTUALIZATION PERFORMANCE COMPARISON REPORT\n");
    fprintf(f, "================================================================================\n\n");

    fprintf(f, "Baseline System:  %s v%s\n", baseline->system_name, baseline->system_version);
    fprintf(f, "Improved System:  %s v%s\n", improved->system_name, improved->system_version);
    fprintf(f, "Timestamp:        %s\n\n", date_str);

    /* Overall comparison */
    fprintf(f, "================================================================================\n");
    fprintf(f, "                           SCORE COMPARISON\n");
    fprintf(f, "================================================================================\n\n");

    fprintf(f, "  +------------------+------------+------------+------------+\n");
    fprintf(f, "  | Metric           | %-10s | %-10s | Change     |\n",
            baseline->system_name, improved->system_name);
    fprintf(f, "  +------------------+------------+------------+------------+\n");

    double overall_change = (improved->overall_score - baseline->overall_score) * 100.0;
    double overhead_change = (improved->overhead_score - baseline->overhead_score) * 100.0;
    double isolation_change = (improved->isolation_score - baseline->isolation_score) * 100.0;
    double llm_change = (improved->llm_score - baseline->llm_score) * 100.0;
    double mig_change = improved->mig_parity_percent - baseline->mig_parity_percent;

    fprintf(f, "  | Overall Score    | %9.1f%% | %9.1f%% | %+9.1f%% |\n",
            baseline->overall_score * 100.0, improved->overall_score * 100.0, overall_change);
    fprintf(f, "  | Overhead Score   | %9.1f%% | %9.1f%% | %+9.1f%% |\n",
            baseline->overhead_score * 100.0, improved->overhead_score * 100.0, overhead_change);
    fprintf(f, "  | Isolation Score  | %9.1f%% | %9.1f%% | %+9.1f%% |\n",
            baseline->isolation_score * 100.0, improved->isolation_score * 100.0, isolation_change);
    fprintf(f, "  | LLM Score        | %9.1f%% | %9.1f%% | %+9.1f%% |\n",
            baseline->llm_score * 100.0, improved->llm_score * 100.0, llm_change);
    fprintf(f, "  | MIG Parity       | %9.1f%% | %9.1f%% | %+9.1f%% |\n",
            baseline->mig_parity_percent, improved->mig_parity_percent, mig_change);
    fprintf(f, "  +------------------+------------+------------+------------+\n\n");

    /* Detailed metric comparison */
    fprintf(f, "================================================================================\n");
    fprintf(f, "                        DETAILED METRIC COMPARISON\n");
    fprintf(f, "================================================================================\n\n");

    fprintf(f, "  %-8s %-25s %10s %10s %10s %s\n",
            "ID", "Metric", "Baseline", "Improved", "Change%", "Result");
    fprintf(f, "  %-8s %-25s %10s %10s %10s %s\n",
            "--------", "-------------------------", "----------",
            "----------", "----------", "------");

    /* Match metrics between baseline and improved */
    for (int i = 0; i < baseline->num_metrics; i++) {
        const metric_result_t *base_m = &baseline->metrics[i];

        /* Find matching metric in improved */
        const metric_result_t *impr_m = NULL;
        for (int j = 0; j < improved->num_metrics; j++) {
            if (strcmp(baseline->metrics[i].metric_id,
                       improved->metrics[j].metric_id) == 0) {
                impr_m = &improved->metrics[j];
                break;
            }
        }

        if (impr_m == NULL || !base_m->success || !impr_m->success) {
            fprintf(f, "  %-8s %-25.25s %10s %10s %10s %s\n",
                    base_m->metric_id, base_m->name, "-", "-", "-", "N/A");
            continue;
        }

        /* Calculate improvement - for latency metrics, lower is better */
        double change_percent;
        const char *result_str;
        bool lower_is_better = (strstr(base_m->unit, "us") != NULL ||
                               strstr(base_m->unit, "ms") != NULL ||
                               strstr(base_m->name, "Latency") != NULL ||
                               strstr(base_m->name, "Overhead") != NULL);

        if (base_m->stats.mean == 0.0) {
            change_percent = 0.0;
        } else {
            change_percent = ((impr_m->stats.mean - base_m->stats.mean) /
                             base_m->stats.mean) * 100.0;
        }

        if (lower_is_better) {
            /* For latency: negative change = improvement */
            if (change_percent < -5.0) result_str = "BETTER";
            else if (change_percent > 5.0) result_str = "WORSE";
            else result_str = "SAME";
        } else {
            /* For throughput/scores: positive change = improvement */
            if (change_percent > 5.0) result_str = "BETTER";
            else if (change_percent < -5.0) result_str = "WORSE";
            else result_str = "SAME";
        }

        fprintf(f, "  %-8s %-25.25s %10.3f %10.3f %+9.1f%% %s\n",
                base_m->metric_id, base_m->name,
                base_m->stats.mean, impr_m->stats.mean,
                change_percent, result_str);
    }

    fprintf(f, "\n");

    /* Summary */
    int improvements = 0, regressions = 0, unchanged = 0;
    for (int i = 0; i < baseline->num_metrics; i++) {
        const metric_result_t *base_m = &baseline->metrics[i];
        const metric_result_t *impr_m = NULL;

        for (int j = 0; j < improved->num_metrics; j++) {
            if (strcmp(baseline->metrics[i].metric_id,
                       improved->metrics[j].metric_id) == 0) {
                impr_m = &improved->metrics[j];
                break;
            }
        }

        if (impr_m == NULL || !base_m->success || !impr_m->success) continue;

        double change = (impr_m->stats.mean - base_m->stats.mean) / base_m->stats.mean;
        bool lower_is_better = (strstr(base_m->unit, "us") != NULL ||
                               strstr(base_m->unit, "ms") != NULL);

        if (lower_is_better) change = -change;

        if (change > 0.05) improvements++;
        else if (change < -0.05) regressions++;
        else unchanged++;
    }

    fprintf(f, "================================================================================\n");
    fprintf(f, "                              SUMMARY\n");
    fprintf(f, "================================================================================\n\n");
    fprintf(f, "  Metrics Improved:  %d\n", improvements);
    fprintf(f, "  Metrics Regressed: %d\n", regressions);
    fprintf(f, "  Metrics Unchanged: %d\n\n", unchanged);

    if (overall_change > 5.0) {
        fprintf(f, "  VERDICT: %s shows SIGNIFICANT IMPROVEMENT over %s\n",
                improved->system_name, baseline->system_name);
    } else if (overall_change < -5.0) {
        fprintf(f, "  VERDICT: %s shows REGRESSION compared to %s\n",
                improved->system_name, baseline->system_name);
    } else {
        fprintf(f, "  VERDICT: No significant difference between systems\n");
    }

    fprintf(f, "================================================================================\n");

    fclose(f);
    printf("Comparison report saved: %s\n", file_path);
    return 0;
}

/*
 * ============================================================================
 * Console Output
 * ============================================================================
 */

void report_print_summary(const benchmark_result_t *result) {
    printf("\n");
    printf("================================================================================\n");
    printf("  BENCHMARK SUMMARY: %s v%s\n", result->system_name, result->system_version);
    printf("================================================================================\n\n");

    printf("  Overall Score:    %.1f%% - %s\n",
           result->overall_score * 100.0, get_score_grade(result->overall_score));
    printf("  MIG Parity:       %.1f%%\n\n", result->mig_parity_percent);

    printf("  Category Scores:\n");
    printf("    Overhead:       %.1f%%\n", result->overhead_score * 100.0);
    printf("    Isolation:      %.1f%%\n", result->isolation_score * 100.0);
    printf("    LLM:            %.1f%%\n", result->llm_score * 100.0);

    printf("\n  Duration: %.2f seconds\n", result->total_duration_seconds);
    printf("================================================================================\n\n");
}

void report_print_metric(const metric_result_t *m) {
    if (!m->success) {
        printf("  [FAIL] %s (%s): %s\n", m->metric_id, m->name, m->error_message);
        return;
    }

    printf("  [OK]   %s (%s)\n", m->metric_id, m->name);
    printf("         Mean: %.3f %s (std: %.3f)\n", m->stats.mean, m->unit, m->stats.std_dev);
    printf("         MIG Expected: %.3f %s (gap: %.1f%%)\n",
           m->mig_expected, m->unit, m->mig_gap_percent);
}

/*
 * ============================================================================
 * Save All Reports
 * ============================================================================
 */

int report_save_all(const benchmark_result_t *result, const char *output_dir) {
    int ret = 0;

    if (report_save_json(result, output_dir) != 0) ret = -1;
    if (report_save_csv(result, output_dir) != 0) ret = -1;
    if (report_save_text(result, output_dir) != 0) ret = -1;

    return ret;
}

/*
 * ============================================================================
 * Result Aggregation
 * ============================================================================
 */

void report_calculate_scores(benchmark_result_t *result) {
    double overhead_sum = 0.0, overhead_count = 0.0;
    double isolation_sum = 0.0, isolation_count = 0.0;
    double llm_sum = 0.0, llm_count = 0.0;

    for (int i = 0; i < result->num_metrics; i++) {
        metric_result_t *m = &result->metrics[i];
        if (!m->success) continue;

        if (strncmp(m->metric_id, "OH-", 3) == 0) {
            overhead_sum += m->normalized_score;
            overhead_count += 1.0;
        } else if (strncmp(m->metric_id, "IS-", 3) == 0) {
            isolation_sum += m->normalized_score;
            isolation_count += 1.0;
        } else if (strncmp(m->metric_id, "LLM-", 4) == 0) {
            llm_sum += m->normalized_score;
            llm_count += 1.0;
        }
    }

    result->overhead_score = (overhead_count > 0) ? (overhead_sum / overhead_count) : 0.0;
    result->isolation_score = (isolation_count > 0) ? (isolation_sum / isolation_count) : 0.0;
    result->llm_score = (llm_count > 0) ? (llm_sum / llm_count) : 0.0;

    /* Weighted overall: Overhead 30%, Isolation 30%, LLM 40% */
    result->overall_score = (result->overhead_score * 0.30 +
                            result->isolation_score * 0.30 +
                            result->llm_score * 0.40);

    /* MIG parity is average of all normalized scores */
    double total_sum = overhead_sum + isolation_sum + llm_sum;
    double total_count = overhead_count + isolation_count + llm_count;
    result->mig_parity_percent = (total_count > 0) ?
                                 ((total_sum / total_count) * 100.0) : 0.0;
}
