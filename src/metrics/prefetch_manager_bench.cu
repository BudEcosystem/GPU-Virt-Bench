/**
 * Prefetch Manager Benchmarks
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include "../../../bud_fcsp/src/uvm/prefetch_manager.h"
#include "../../../bud_fcsp/src/uvm/uvm_types.h"

static inline uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static uvm_alloc_metadata_t* create_test_allocation(size_t size) {
    uvm_alloc_metadata_t *alloc = (uvm_alloc_metadata_t*)calloc(1, sizeof(uvm_alloc_metadata_t));
    if (!alloc) return NULL;
    cudaMalloc(&alloc->gpu_ptr, size);
    alloc->cpu_ptr = malloc(size);
    alloc->size = size;
    alloc->location = UVM_LOCATION_CPU;
    alloc->access_count = rand() % 100;
    alloc->last_access_ns = get_time_ns() - (rand() % 1000000000);
    alloc->prefetch_count = rand() % 20;
    return alloc;
}

static void free_test_allocation(uvm_alloc_metadata_t *alloc) {
    if (!alloc) return;
    if (alloc->gpu_ptr) cudaFree(alloc->gpu_ptr);
    if (alloc->cpu_ptr) free(alloc->cpu_ptr);
    free(alloc);
}

extern "C" void run_prefetch_manager_benchmarks(void) {
    printf("\n=================================================\n");
    printf("Prefetch Manager Benchmarks\n");
    printf("=================================================\n\n");

    // Benchmark 1: Score Calculation Performance
    {
        const int num_allocs = 1000;
        const int iters = 10000;

        uvm_prefetch_manager_t *manager = uvm_prefetch_manager_init(0, PREFETCH_MODE_MODERATE);
        uvm_alloc_metadata_t **allocs = (uvm_alloc_metadata_t**)calloc(num_allocs, sizeof(uvm_alloc_metadata_t*));

        for (int i = 0; i < num_allocs; i++) {
            allocs[i] = create_test_allocation(4 * 1024 * 1024);
        }

        uint64_t start = get_time_ns();
        for (int i = 0; i < iters; i++) {
            uvm_prefetch_calculate_score(manager, allocs[i % num_allocs]);
        }
        uint64_t end = get_time_ns();

        double avg_time_ns = (end - start) / (double)iters;

        printf("%-45s %.2f ns/calculation\n", "Multi-Factor Score Calculation", avg_time_ns);

        for (int i = 0; i < num_allocs; i++) {
            free_test_allocation(allocs[i]);
        }
        free(allocs);
        uvm_prefetch_manager_shutdown(manager);
    }

    // Benchmark 2: Candidate Selection Performance
    {
        const int sizes[] = {100, 500, 1000, 5000};
        const int iters = 1000;

        printf("%-45s ", "Candidate Selection Scalability");
        for (int s = 0; s < 4; s++) {
            int num_allocs = sizes[s];

            uvm_prefetch_manager_t *manager = uvm_prefetch_manager_init(0, PREFETCH_MODE_MODERATE);
            uvm_alloc_metadata_t **allocs = (uvm_alloc_metadata_t**)calloc(num_allocs, sizeof(uvm_alloc_metadata_t*));

            for (int i = 0; i < num_allocs; i++) {
                allocs[i] = create_test_allocation(1 * 1024 * 1024);
                uvm_prefetch_register_candidate(manager, allocs[i]);
            }

            uint64_t start = get_time_ns();
            for (int i = 0; i < iters; i++) {
                uvm_prefetch_candidate_t *candidates = NULL;
                int count = uvm_prefetch_select_candidates(manager, &candidates, 10);
                if (candidates) free(candidates);
            }
            uint64_t end = get_time_ns();

            double avg_time_us = (end - start) / (1000.0 * iters);
            printf("%.2fus@%d ", avg_time_us, num_allocs);

            for (int i = 0; i < num_allocs; i++) {
                uvm_prefetch_unregister_candidate(manager, allocs[i]);
                free_test_allocation(allocs[i]);
            }
            free(allocs);
            uvm_prefetch_manager_shutdown(manager);
        }
        printf("\n");
    }

    // Benchmark 3: Mode Switching Performance
    {
        const int num_modes = 4;
        const int switches_per_mode = 1000;
        const int total_switches = num_modes * switches_per_mode;

        uvm_prefetch_manager_t *manager = uvm_prefetch_manager_init(0, PREFETCH_MODE_MODERATE);

        prefetch_mode_t modes[] = {
            PREFETCH_MODE_CONSERVATIVE,
            PREFETCH_MODE_MODERATE,
            PREFETCH_MODE_AGGRESSIVE,
            PREFETCH_MODE_ADAPTIVE
        };

        uint64_t start = get_time_ns();
        for (int i = 0; i < total_switches; i++) {
            uvm_prefetch_set_mode(manager, modes[i % num_modes]);
        }
        uint64_t end = get_time_ns();

        double avg_time_ns = (end - start) / (double)total_switches;

        printf("%-45s %.2f ns/switch\n", "Mode Switching Overhead", avg_time_ns);

        uvm_prefetch_manager_shutdown(manager);
    }

    // Benchmark 4: Prefetch Trigger Performance (Conservative vs Aggressive)
    {
        const int num_allocs = 100;

        // Conservative mode
        uvm_prefetch_manager_t *manager_cons = uvm_prefetch_manager_init(0, PREFETCH_MODE_CONSERVATIVE);
        uvm_alloc_metadata_t **allocs_cons = (uvm_alloc_metadata_t**)calloc(num_allocs, sizeof(uvm_alloc_metadata_t*));

        for (int i = 0; i < num_allocs; i++) {
            allocs_cons[i] = create_test_allocation(1 * 1024 * 1024);
            allocs_cons[i]->access_count = 100 + rand() % 400;  // High access counts
            allocs_cons[i]->last_access_ns = get_time_ns() - (rand() % 50000000);  // Recent
        }

        uint64_t start_cons = get_time_ns();
        int triggered_cons = 0;
        for (int i = 0; i < num_allocs; i++) {
            if (uvm_prefetch_trigger(manager_cons, allocs_cons[i])) {
                triggered_cons++;
            }
        }
        cudaDeviceSynchronize();
        uint64_t end_cons = get_time_ns();

        // Aggressive mode
        uvm_prefetch_manager_t *manager_agg = uvm_prefetch_manager_init(0, PREFETCH_MODE_AGGRESSIVE);
        uvm_alloc_metadata_t **allocs_agg = (uvm_alloc_metadata_t**)calloc(num_allocs, sizeof(uvm_alloc_metadata_t*));

        for (int i = 0; i < num_allocs; i++) {
            allocs_agg[i] = create_test_allocation(1 * 1024 * 1024);
            allocs_agg[i]->access_count = rand() % 50;  // Lower access counts
            allocs_agg[i]->last_access_ns = get_time_ns() - (rand() % 500000000);  // Less recent
        }

        uint64_t start_agg = get_time_ns();
        int triggered_agg = 0;
        for (int i = 0; i < num_allocs; i++) {
            if (uvm_prefetch_trigger(manager_agg, allocs_agg[i])) {
                triggered_agg++;
            }
        }
        cudaDeviceSynchronize();
        uint64_t end_agg = get_time_ns();

        double time_cons_us = (end_cons - start_cons) / 1000.0;
        double time_agg_us = (end_agg - start_agg) / 1000.0;

        printf("%-45s %d/%d (%.1f%%) in %.2f us\n",
               "Conservative Mode Prefetch Rate",
               triggered_cons, num_allocs,
               (triggered_cons * 100.0) / num_allocs,
               time_cons_us);

        printf("%-45s %d/%d (%.1f%%) in %.2f us\n",
               "Aggressive Mode Prefetch Rate",
               triggered_agg, num_allocs,
               (triggered_agg * 100.0) / num_allocs,
               time_agg_us);

        for (int i = 0; i < num_allocs; i++) {
            free_test_allocation(allocs_cons[i]);
            free_test_allocation(allocs_agg[i]);
        }
        free(allocs_cons);
        free(allocs_agg);
        uvm_prefetch_manager_shutdown(manager_cons);
        uvm_prefetch_manager_shutdown(manager_agg);
    }

    // Benchmark 5: Batch Prefetch Throughput
    {
        const int num_allocs = 1000;
        const int batch_sizes[] = {1, 5, 10, 20, 50};

        printf("%-45s ", "Batch Prefetch Throughput");
        for (int b = 0; b < 5; b++) {
            int batch_size = batch_sizes[b];

            uvm_prefetch_manager_t *manager = uvm_prefetch_manager_init(0, PREFETCH_MODE_AGGRESSIVE);
            uvm_alloc_metadata_t **allocs = (uvm_alloc_metadata_t**)calloc(num_allocs, sizeof(uvm_alloc_metadata_t*));

            for (int i = 0; i < num_allocs; i++) {
                allocs[i] = create_test_allocation(512 * 1024);  // 512KB each
                allocs[i]->access_count = 100;
                allocs[i]->last_access_ns = get_time_ns() - 10000000;
                uvm_prefetch_register_candidate(manager, allocs[i]);
            }

            uint64_t start = get_time_ns();
            int total_prefetched = uvm_prefetch_trigger_batch(manager, batch_size);
            cudaDeviceSynchronize();
            uint64_t end = get_time_ns();

            double time_us = (end - start) / 1000.0;
            double throughput_gbps = (total_prefetched * 512 * 1024) / (time_us * 1000.0 / 8.0);  // GB/s

            printf("%.2fGB/s@%d ", throughput_gbps, batch_size);

            for (int i = 0; i < num_allocs; i++) {
                uvm_prefetch_unregister_candidate(manager, allocs[i]);
                free_test_allocation(allocs[i]);
            }
            free(allocs);
            uvm_prefetch_manager_shutdown(manager);
        }
        printf("\n");
    }

    // Benchmark 6: Adaptive Aggressiveness Overhead
    {
        const int iters = 10000;

        uvm_prefetch_manager_t *manager = uvm_prefetch_manager_init(0, PREFETCH_MODE_ADAPTIVE);

        // Simulate varying accuracy
        pthread_mutex_lock(&manager->lock);
        manager->successful_prefetches = 80;
        manager->wasted_prefetches = 20;

        uint64_t start = get_time_ns();
        for (int i = 0; i < iters; i++) {
            uvm_prefetch_adapt_aggressiveness(manager);
        }
        uint64_t end = get_time_ns();
        pthread_mutex_unlock(&manager->lock);

        double avg_time_ns = (end - start) / (double)iters;

        printf("%-45s %.2f ns/adaptation\n", "Adaptive Aggressiveness Overhead", avg_time_ns);

        uvm_prefetch_manager_shutdown(manager);
    }

    // Benchmark 7: Register/Unregister Overhead
    {
        const int num_allocs = 1000;
        const int iters = 1000;

        uvm_prefetch_manager_t *manager = uvm_prefetch_manager_init(0, PREFETCH_MODE_MODERATE);
        uvm_alloc_metadata_t **allocs = (uvm_alloc_metadata_t**)calloc(num_allocs, sizeof(uvm_alloc_metadata_t*));

        for (int i = 0; i < num_allocs; i++) {
            allocs[i] = create_test_allocation(1 * 1024 * 1024);
        }

        // Register benchmark
        uint64_t start = get_time_ns();
        for (int i = 0; i < iters; i++) {
            uvm_prefetch_register_candidate(manager, allocs[i % num_allocs]);
        }
        uint64_t end = get_time_ns();
        double reg_time_ns = (end - start) / (double)iters;

        // Unregister benchmark
        start = get_time_ns();
        for (int i = 0; i < iters; i++) {
            uvm_prefetch_unregister_candidate(manager, allocs[i % num_allocs]);
        }
        end = get_time_ns();
        double unreg_time_ns = (end - start) / (double)iters;

        printf("%-45s %.2f ns/register\n", "Candidate Register Overhead", reg_time_ns);
        printf("%-45s %.2f ns/unregister\n", "Candidate Unregister Overhead", unreg_time_ns);

        for (int i = 0; i < num_allocs; i++) {
            free_test_allocation(allocs[i]);
        }
        free(allocs);
        uvm_prefetch_manager_shutdown(manager);
    }

    printf("\n");
}
