/*
 * GPU Virtualization Performance Evaluation Tool
 * Statistical analysis utilities
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "include/benchmark.h"

/*
 * ============================================================================
 * Comparison function for qsort
 * ============================================================================
 */

static int compare_double(const void *a, const void *b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

/*
 * ============================================================================
 * Basic Statistics
 * ============================================================================
 */

void stats_calculate(const double *values, uint64_t count, statistics_t *stats) {
    if (count == 0 || values == NULL || stats == NULL) {
        memset(stats, 0, sizeof(statistics_t));
        return;
    }

    stats->count = count;

    /* First pass: min, max, sum for mean */
    double sum = 0.0;
    stats->min = DBL_MAX;
    stats->max = -DBL_MAX;

    for (uint64_t i = 0; i < count; i++) {
        sum += values[i];
        if (values[i] < stats->min) stats->min = values[i];
        if (values[i] > stats->max) stats->max = values[i];
    }

    stats->mean = sum / (double)count;

    /* Second pass: variance for std_dev */
    double variance_sum = 0.0;
    for (uint64_t i = 0; i < count; i++) {
        double diff = values[i] - stats->mean;
        variance_sum += diff * diff;
    }
    stats->std_dev = sqrt(variance_sum / (double)count);

    /* Create sorted copy for percentiles */
    double *sorted = (double*)malloc(sizeof(double) * count);
    if (sorted == NULL) {
        stats->median = stats->mean;
        stats->p50 = stats->mean;
        stats->p90 = stats->max;
        stats->p95 = stats->max;
        stats->p99 = stats->max;
        stats->p999 = stats->max;
        return;
    }

    memcpy(sorted, values, sizeof(double) * count);
    qsort(sorted, count, sizeof(double), compare_double);

    /* Percentiles */
    stats->p50 = sorted[(uint64_t)(count * 0.50)];
    stats->median = stats->p50;
    stats->p90 = sorted[(uint64_t)(count * 0.90)];
    stats->p95 = sorted[(uint64_t)(count * 0.95)];
    stats->p99 = sorted[(uint64_t)(count * 0.99)];

    if (count >= 1000) {
        stats->p999 = sorted[(uint64_t)(count * 0.999)];
    } else {
        stats->p999 = stats->max;
    }

    free(sorted);
}

/*
 * ============================================================================
 * Jain's Fairness Index
 *
 * Formula: (sum(x_i))^2 / (n * sum(x_i^2))
 * Range: 1/n (most unfair) to 1.0 (perfectly fair)
 * ============================================================================
 */

double stats_jains_fairness(const double *values, int count) {
    if (count <= 0 || values == NULL) return 0.0;
    if (count == 1) return 1.0;

    double sum = 0.0;
    double sum_sq = 0.0;

    for (int i = 0; i < count; i++) {
        sum += values[i];
        sum_sq += values[i] * values[i];
    }

    if (sum_sq == 0.0) return 1.0;

    double fairness = (sum * sum) / ((double)count * sum_sq);
    return fairness;
}

/*
 * ============================================================================
 * Coefficient of Variation (CV)
 *
 * Formula: std_dev / mean
 * Lower = more consistent
 * ============================================================================
 */

double stats_coefficient_of_variation(const statistics_t *stats) {
    if (stats == NULL || stats->mean == 0.0) return 0.0;
    return stats->std_dev / fabs(stats->mean);
}

/*
 * ============================================================================
 * Throughput Calculations
 * ============================================================================
 */

double stats_throughput_ops_per_sec(uint64_t operations, double elapsed_seconds) {
    if (elapsed_seconds <= 0.0) return 0.0;
    return (double)operations / elapsed_seconds;
}

double stats_throughput_bytes_per_sec(uint64_t bytes, double elapsed_seconds) {
    if (elapsed_seconds <= 0.0) return 0.0;
    return (double)bytes / elapsed_seconds;
}

double stats_throughput_gb_per_sec(uint64_t bytes, double elapsed_seconds) {
    return stats_throughput_bytes_per_sec(bytes, elapsed_seconds) / (1024.0 * 1024.0 * 1024.0);
}

/*
 * ============================================================================
 * Performance Comparison
 * ============================================================================
 */

double stats_speedup(double baseline_time, double comparison_time) {
    if (comparison_time <= 0.0) return 0.0;
    return baseline_time / comparison_time;
}

double stats_improvement_percent(double baseline, double improved) {
    if (baseline <= 0.0) return 0.0;
    return ((baseline - improved) / baseline) * 100.0;
}

double stats_degradation_percent(double baseline, double degraded) {
    if (baseline <= 0.0) return 0.0;
    return ((degraded - baseline) / baseline) * 100.0;
}

/*
 * ============================================================================
 * Confidence Interval Calculation
 * Using t-distribution for small samples
 * ============================================================================
 */

/* T-table values for 95% confidence (two-tailed) */
static const double T_VALUES_95[] = {
    12.706,  /* n=2 */
    4.303,   /* n=3 */
    3.182,   /* n=4 */
    2.776,   /* n=5 */
    2.571,   /* n=6 */
    2.447,   /* n=7 */
    2.365,   /* n=8 */
    2.306,   /* n=9 */
    2.262,   /* n=10 */
    2.228,   /* n=11 */
    2.201,   /* n=12 */
    2.179,   /* n=13 */
    2.160,   /* n=14 */
    2.145,   /* n=15 */
    2.131,   /* n=16 */
    2.120,   /* n=17 */
    2.110,   /* n=18 */
    2.101,   /* n=19 */
    2.093,   /* n=20 */
    2.086,   /* n=21-25 */
    2.042,   /* n=26-30 */
    2.021,   /* n=31-40 */
    2.000,   /* n=41-60 */
    1.984,   /* n=61-120 */
    1.960,   /* n>120 (approx Z) */
};

static double get_t_value_95(int n) {
    if (n <= 1) return 0.0;
    if (n <= 20) return T_VALUES_95[n - 2];
    if (n <= 25) return T_VALUES_95[19];
    if (n <= 30) return T_VALUES_95[20];
    if (n <= 40) return T_VALUES_95[21];
    if (n <= 60) return T_VALUES_95[22];
    if (n <= 120) return T_VALUES_95[23];
    return T_VALUES_95[24];
}

void stats_confidence_interval_95(const statistics_t *stats,
                                  double *lower, double *upper) {
    if (stats == NULL || stats->count < 2) {
        *lower = *upper = 0.0;
        return;
    }

    double t = get_t_value_95((int)stats->count);
    double margin = t * stats->std_dev / sqrt((double)stats->count);

    *lower = stats->mean - margin;
    *upper = stats->mean + margin;
}

/*
 * ============================================================================
 * Outlier Detection (IQR method)
 * ============================================================================
 */

int stats_detect_outliers(const double *values, uint64_t count,
                         double *outliers, int max_outliers) {
    if (count < 4 || values == NULL) return 0;

    /* Create sorted copy */
    double *sorted = (double*)malloc(sizeof(double) * count);
    if (sorted == NULL) return 0;

    memcpy(sorted, values, sizeof(double) * count);
    qsort(sorted, count, sizeof(double), compare_double);

    /* Calculate quartiles */
    double q1 = sorted[count / 4];
    double q3 = sorted[(count * 3) / 4];
    double iqr = q3 - q1;

    double lower_bound = q1 - 1.5 * iqr;
    double upper_bound = q3 + 1.5 * iqr;

    /* Find outliers */
    int outlier_count = 0;
    for (uint64_t i = 0; i < count && outlier_count < max_outliers; i++) {
        if (values[i] < lower_bound || values[i] > upper_bound) {
            outliers[outlier_count++] = values[i];
        }
    }

    free(sorted);
    return outlier_count;
}

/*
 * ============================================================================
 * Remove Outliers and Recalculate
 * ============================================================================
 */

void stats_calculate_without_outliers(const double *values, uint64_t count,
                                      statistics_t *stats) {
    if (count < 4) {
        stats_calculate(values, count, stats);
        return;
    }

    /* Create sorted copy */
    double *sorted = (double*)malloc(sizeof(double) * count);
    if (sorted == NULL) {
        stats_calculate(values, count, stats);
        return;
    }

    memcpy(sorted, values, sizeof(double) * count);
    qsort(sorted, count, sizeof(double), compare_double);

    /* Calculate quartiles */
    double q1 = sorted[count / 4];
    double q3 = sorted[(count * 3) / 4];
    double iqr = q3 - q1;

    double lower_bound = q1 - 1.5 * iqr;
    double upper_bound = q3 + 1.5 * iqr;

    /* Create filtered array */
    double *filtered = (double*)malloc(sizeof(double) * count);
    uint64_t filtered_count = 0;

    for (uint64_t i = 0; i < count; i++) {
        if (values[i] >= lower_bound && values[i] <= upper_bound) {
            filtered[filtered_count++] = values[i];
        }
    }

    /* Calculate stats on filtered data */
    stats_calculate(filtered, filtered_count, stats);

    free(sorted);
    free(filtered);
}

/*
 * ============================================================================
 * Statistical Significance Testing
 * Welch's t-test for comparing two samples
 * ============================================================================
 */

typedef struct {
    double t_statistic;
    double degrees_of_freedom;
    bool significant_at_05;
    bool significant_at_01;
} ttest_result_t;

void stats_welch_ttest(const statistics_t *sample1, const statistics_t *sample2,
                       ttest_result_t *result) {
    if (sample1 == NULL || sample2 == NULL || result == NULL) return;
    if (sample1->count < 2 || sample2->count < 2) return;

    double n1 = (double)sample1->count;
    double n2 = (double)sample2->count;
    double var1 = sample1->std_dev * sample1->std_dev;
    double var2 = sample2->std_dev * sample2->std_dev;

    /* t statistic */
    double se = sqrt(var1/n1 + var2/n2);
    if (se == 0.0) {
        result->t_statistic = 0.0;
        result->significant_at_05 = false;
        result->significant_at_01 = false;
        return;
    }

    result->t_statistic = (sample1->mean - sample2->mean) / se;

    /* Welch-Satterthwaite degrees of freedom */
    double term1 = var1/n1;
    double term2 = var2/n2;
    double numerator = (term1 + term2) * (term1 + term2);
    double denominator = (term1*term1)/(n1-1) + (term2*term2)/(n2-1);
    result->degrees_of_freedom = numerator / denominator;

    /* Approximate significance (rough critical values) */
    double abs_t = fabs(result->t_statistic);
    result->significant_at_05 = (abs_t > 1.96);  /* Approx for large df */
    result->significant_at_01 = (abs_t > 2.576);
}

/*
 * ============================================================================
 * Histogram Calculation
 * ============================================================================
 */

typedef struct {
    double *bin_edges;
    int *bin_counts;
    int num_bins;
    double bin_width;
} histogram_t;

histogram_t* stats_histogram_create(const double *values, uint64_t count, int num_bins) {
    if (count == 0 || num_bins <= 0) return NULL;

    histogram_t *hist = (histogram_t*)malloc(sizeof(histogram_t));
    if (hist == NULL) return NULL;

    hist->bin_edges = (double*)malloc(sizeof(double) * (num_bins + 1));
    hist->bin_counts = (int*)calloc(num_bins, sizeof(int));
    hist->num_bins = num_bins;

    if (hist->bin_edges == NULL || hist->bin_counts == NULL) {
        free(hist->bin_edges);
        free(hist->bin_counts);
        free(hist);
        return NULL;
    }

    /* Find min/max */
    double min_val = values[0], max_val = values[0];
    for (uint64_t i = 1; i < count; i++) {
        if (values[i] < min_val) min_val = values[i];
        if (values[i] > max_val) max_val = values[i];
    }

    /* Calculate bin edges */
    hist->bin_width = (max_val - min_val) / num_bins;
    for (int i = 0; i <= num_bins; i++) {
        hist->bin_edges[i] = min_val + i * hist->bin_width;
    }

    /* Count values in each bin */
    for (uint64_t i = 0; i < count; i++) {
        int bin = (int)((values[i] - min_val) / hist->bin_width);
        if (bin >= num_bins) bin = num_bins - 1;
        if (bin < 0) bin = 0;
        hist->bin_counts[bin]++;
    }

    return hist;
}

void stats_histogram_destroy(histogram_t *hist) {
    if (hist) {
        free(hist->bin_edges);
        free(hist->bin_counts);
        free(hist);
    }
}

void stats_histogram_print(const histogram_t *hist) {
    if (hist == NULL) return;

    int max_count = 0;
    for (int i = 0; i < hist->num_bins; i++) {
        if (hist->bin_counts[i] > max_count) max_count = hist->bin_counts[i];
    }

    printf("Histogram:\n");
    for (int i = 0; i < hist->num_bins; i++) {
        printf("[%8.2f - %8.2f): %6d |",
               hist->bin_edges[i], hist->bin_edges[i+1], hist->bin_counts[i]);

        int bar_len = (max_count > 0) ? (hist->bin_counts[i] * 40 / max_count) : 0;
        for (int j = 0; j < bar_len; j++) printf("#");
        printf("\n");
    }
}
