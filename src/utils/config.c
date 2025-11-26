/*
 * GPU Virtualization Performance Evaluation Tool
 * Configuration file parser (INI-style format)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>

#include "include/benchmark.h"

/*
 * ============================================================================
 * Configuration File Format (INI-style):
 *
 * [general]
 * iterations = 100
 * warmup = 10
 * verbose = false
 * output_dir = ./benchmarks
 *
 * [overhead]
 * enabled = true
 * kernel_launch_iterations = 1000
 * allocation_sizes = 1024,65536,1048576
 *
 * [isolation]
 * enabled = true
 * contention_thread_count = 8
 *
 * [llm]
 * enabled = true
 * hidden_sizes = 768,1024,2048,4096
 * batch_sizes = 1,2,4,8,16,32
 *
 * ============================================================================
 */

#define CONFIG_MAX_LINE_LENGTH 1024
#define CONFIG_MAX_SECTION_LENGTH 64
#define CONFIG_MAX_KEY_LENGTH 64
#define CONFIG_MAX_VALUE_LENGTH 512

/* Trim whitespace from both ends of a string */
static char* trim(char *str) {
    char *end;

    /* Trim leading space */
    while (isspace((unsigned char)*str)) str++;

    if (*str == 0) return str;

    /* Trim trailing space */
    end = str + strlen(str) - 1;
    while (end > str && isspace((unsigned char)*end)) end--;

    /* Write new null terminator */
    *(end + 1) = '\0';

    return str;
}

/* Parse a boolean value */
static bool parse_bool(const char *value) {
    if (strcasecmp(value, "true") == 0 ||
        strcasecmp(value, "yes") == 0 ||
        strcasecmp(value, "1") == 0 ||
        strcasecmp(value, "on") == 0) {
        return true;
    }
    return false;
}

/* Parse a comma-separated list of integers into an array */
static int parse_int_list(const char *value, int *array, int max_count) {
    char *copy = strdup(value);
    if (!copy) return 0;

    int count = 0;
    char *token = strtok(copy, ",");

    while (token && count < max_count) {
        array[count++] = atoi(trim(token));
        token = strtok(NULL, ",");
    }

    free(copy);
    return count;
}

/* Parse a comma-separated list of size_t values */
static int parse_size_list(const char *value, size_t *array, int max_count) {
    char *copy = strdup(value);
    if (!copy) return 0;

    int count = 0;
    char *token = strtok(copy, ",");

    while (token && count < max_count) {
        array[count++] = (size_t)strtoull(trim(token), NULL, 10);
        token = strtok(NULL, ",");
    }

    free(copy);
    return count;
}

/*
 * ============================================================================
 * Configuration Loading
 * ============================================================================
 */

int bench_load_config(const char *config_file, benchmark_config_t *config) {
    FILE *f = fopen(config_file, "r");
    if (!f) {
        LOG_WARN("Could not open config file: %s", config_file);
        return -1;
    }

    char line[CONFIG_MAX_LINE_LENGTH];
    char current_section[CONFIG_MAX_SECTION_LENGTH] = "";
    int line_num = 0;

    while (fgets(line, sizeof(line), f)) {
        line_num++;
        char *trimmed = trim(line);

        /* Skip empty lines and comments */
        if (*trimmed == '\0' || *trimmed == '#' || *trimmed == ';') {
            continue;
        }

        /* Check for section header */
        if (*trimmed == '[') {
            char *end = strchr(trimmed, ']');
            if (end) {
                *end = '\0';
                strncpy(current_section, trimmed + 1, CONFIG_MAX_SECTION_LENGTH - 1);
                current_section[CONFIG_MAX_SECTION_LENGTH - 1] = '\0';
            }
            continue;
        }

        /* Parse key = value */
        char *equals = strchr(trimmed, '=');
        if (!equals) {
            LOG_WARN("Config line %d: Invalid format (missing '=')", line_num);
            continue;
        }

        *equals = '\0';
        char *key = trim(trimmed);
        char *value = trim(equals + 1);

        /* Apply configuration based on section */
        if (strcmp(current_section, "general") == 0) {
            if (strcmp(key, "iterations") == 0) {
                config->options.iterations = atoi(value);
            } else if (strcmp(key, "warmup") == 0) {
                config->options.warmup_iterations = atoi(value);
            } else if (strcmp(key, "verbose") == 0) {
                config->options.verbose = parse_bool(value);
            } else if (strcmp(key, "output_dir") == 0) {
                strncpy(config->options.output_dir, value, MAX_PATH_LENGTH - 1);
            } else if (strcmp(key, "timeout") == 0) {
                config->options.timeout_seconds = atoi(value);
            } else if (strcmp(key, "save_raw_data") == 0) {
                config->options.save_raw_data = parse_bool(value);
            }
        }
        else if (strcmp(current_section, "overhead") == 0) {
            if (strcmp(key, "enabled") == 0) {
                config->overhead_enabled = parse_bool(value);
            }
        }
        else if (strcmp(current_section, "isolation") == 0) {
            if (strcmp(key, "enabled") == 0) {
                config->isolation_enabled = parse_bool(value);
            }
        }
        else if (strcmp(current_section, "llm") == 0) {
            if (strcmp(key, "enabled") == 0) {
                config->llm_enabled = parse_bool(value);
            }
        }
        else if (strcmp(current_section, "device") == 0) {
            if (strcmp(key, "device_ids") == 0) {
                config->device_count = parse_int_list(value, config->device_ids, MAX_DEVICES);
            }
        }
    }

    fclose(f);
    LOG_INFO("Loaded configuration from: %s", config_file);
    return 0;
}

/*
 * ============================================================================
 * Configuration Saving
 * ============================================================================
 */

int bench_save_config(const char *config_file, const benchmark_config_t *config) {
    FILE *f = fopen(config_file, "w");
    if (!f) {
        LOG_ERROR("Could not create config file: %s", config_file);
        return -1;
    }

    fprintf(f, "# GPU Virtualization Benchmark Configuration\n");
    fprintf(f, "# Generated automatically\n\n");

    fprintf(f, "[general]\n");
    fprintf(f, "iterations = %d\n", config->options.iterations);
    fprintf(f, "warmup = %d\n", config->options.warmup_iterations);
    fprintf(f, "verbose = %s\n", config->options.verbose ? "true" : "false");
    fprintf(f, "output_dir = %s\n", config->options.output_dir);
    fprintf(f, "timeout = %d\n", config->options.timeout_seconds);
    fprintf(f, "save_raw_data = %s\n", config->options.save_raw_data ? "true" : "false");
    fprintf(f, "\n");

    fprintf(f, "[overhead]\n");
    fprintf(f, "enabled = %s\n", config->overhead_enabled ? "true" : "false");
    fprintf(f, "\n");

    fprintf(f, "[isolation]\n");
    fprintf(f, "enabled = %s\n", config->isolation_enabled ? "true" : "false");
    fprintf(f, "\n");

    fprintf(f, "[llm]\n");
    fprintf(f, "enabled = %s\n", config->llm_enabled ? "true" : "false");
    fprintf(f, "\n");

    fprintf(f, "[device]\n");
    fprintf(f, "# Comma-separated list of device IDs to benchmark\n");
    fprintf(f, "device_ids = 0\n");
    fprintf(f, "\n");

    fclose(f);
    LOG_INFO("Saved configuration to: %s", config_file);
    return 0;
}

/*
 * ============================================================================
 * Default Configuration
 * ============================================================================
 */

void bench_set_defaults(benchmark_config_t *config) {
    memset(config, 0, sizeof(benchmark_config_t));

    strcpy(config->name, "default");

    /* Options */
    config->options.iterations = DEFAULT_ITERATIONS;
    config->options.warmup_iterations = DEFAULT_WARMUP;
    config->options.timeout_seconds = 300;
    config->options.verbose = false;
    config->options.save_raw_data = false;
    strcpy(config->options.output_dir, "./benchmarks");

    /* Enable all metric categories by default */
    config->overhead_enabled = true;
    config->isolation_enabled = true;
    config->llm_enabled = true;

    /* Single device by default */
    config->device_ids[0] = 0;
    config->device_count = 1;
}

/*
 * ============================================================================
 * Configuration Validation
 * ============================================================================
 */

int bench_validate_config(const benchmark_config_t *config) {
    if (config->options.iterations <= 0) {
        LOG_ERROR("Invalid iteration count: %d", config->options.iterations);
        return -1;
    }

    if (config->options.iterations > MAX_ITERATIONS) {
        LOG_ERROR("Iteration count exceeds maximum: %d > %d",
                  config->options.iterations, MAX_ITERATIONS);
        return -1;
    }

    if (config->device_count <= 0) {
        LOG_ERROR("No devices specified");
        return -1;
    }

    if (config->device_count > MAX_DEVICES) {
        LOG_ERROR("Device count exceeds maximum: %d > %d",
                  config->device_count, MAX_DEVICES);
        return -1;
    }

    if (!config->overhead_enabled && !config->isolation_enabled && !config->llm_enabled) {
        LOG_ERROR("No benchmark categories enabled");
        return -1;
    }

    return 0;
}

/*
 * ============================================================================
 * Configuration Printing
 * ============================================================================
 */

void bench_print_config(const benchmark_config_t *config) {
    printf("Configuration:\n");
    printf("  Iterations: %d (warmup: %d)\n",
           config->options.iterations, config->options.warmup_iterations);
    printf("  Output: %s\n", config->options.output_dir);
    printf("  Verbose: %s\n", config->options.verbose ? "yes" : "no");
    printf("  Categories:\n");
    printf("    Overhead: %s\n", config->overhead_enabled ? "enabled" : "disabled");
    printf("    Isolation: %s\n", config->isolation_enabled ? "enabled" : "disabled");
    printf("    LLM: %s\n", config->llm_enabled ? "enabled" : "disabled");
    printf("  Devices: ");
    for (int i = 0; i < config->device_count; i++) {
        printf("%d%s", config->device_ids[i], (i < config->device_count - 1) ? ", " : "");
    }
    printf("\n");
}
