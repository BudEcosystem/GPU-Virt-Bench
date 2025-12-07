/**
 * Eviction Policy Benchmarks
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include "../../../bud_fcsp/src/uvm/eviction_policy.h"
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
    alloc->location = UVM_LOCATION_GPU;
    alloc->access_count = rand() % 100;
    alloc->last_access_ns = get_time_ns() - (rand() % 1000000000);
    return alloc;
}

static void free_test_allocation(uvm_alloc_metadata_t *alloc) {
    if (!alloc) return;
    if (alloc->gpu_ptr) cudaFree(alloc->gpu_ptr);
    if (alloc->cpu_ptr) free(alloc->cpu_ptr);
    free(alloc);
}

extern "C" void run_eviction_policy_benchmarks(void) {
    printf("\n=================================================\n");
    printf("Eviction Policy Benchmarks\n");
    printf("=================================================\n\n");

    // Benchmark 1: LRU Victim Selection Performance
    {
        const int num_allocs = 1000;
        const int iters = 1000;

        uvm_eviction_policy_t *policy = uvm_eviction_policy_init(EVICTION_POLICY_LRU, 0);
        uvm_alloc_metadata_t **allocs = (uvm_alloc_metadata_t**)calloc(num_allocs, sizeof(uvm_alloc_metadata_t*));

        // Create and register allocations
        for (int i = 0; i < num_allocs; i++) {
            allocs[i] = create_test_allocation(4 * 1024 * 1024);
            uvm_eviction_policy_register(policy, allocs[i]);
        }

        // Benchmark victim selection
        uint64_t start = get_time_ns();
        for (int i = 0; i < iters; i++) {
            uvm_eviction_policy_select_victim(policy);
        }
        uint64_t end = get_time_ns();

        double time_us = (end - start) / 1000.0;
        double avg_time_ns = (end - start) / (double)iters;

        printf("%-45s %.2f ns/selection\n", "LRU Victim Selection (1000 allocs)", avg_time_ns);

        for (int i = 0; i < num_allocs; i++) {
            uvm_eviction_policy_unregister(policy, allocs[i]);
            free_test_allocation(allocs[i]);
        }
        free(allocs);
        uvm_eviction_policy_shutdown(policy);
    }

    // Benchmark 2: Access-Aware Victim Selection Performance
    {
        const int num_allocs = 1000;
        const int iters = 1000;

        uvm_eviction_policy_t *policy = uvm_eviction_policy_init(EVICTION_POLICY_ACCESS_AWARE, 0);
        uvm_alloc_metadata_t **allocs = (uvm_alloc_metadata_t**)calloc(num_allocs, sizeof(uvm_alloc_metadata_t*));

        // Create and register allocations
        for (int i = 0; i < num_allocs; i++) {
            allocs[i] = create_test_allocation(4 * 1024 * 1024);
            uvm_eviction_policy_register(policy, allocs[i]);
        }

        // Benchmark victim selection
        uint64_t start = get_time_ns();
        for (int i = 0; i < iters; i++) {
            uvm_eviction_policy_select_victim(policy);
        }
        uint64_t end = get_time_ns();

        double avg_time_ns = (end - start) / (double)iters;

        printf("%-45s %.2f ns/selection\n", "Access-Aware Victim Selection (1000 allocs)", avg_time_ns);

        for (int i = 0; i < num_allocs; i++) {
            uvm_eviction_policy_unregister(policy, allocs[i]);
            free_test_allocation(allocs[i]);
        }
        free(allocs);
        uvm_eviction_policy_shutdown(policy);
    }

    // Benchmark 3: FIFO Victim Selection Performance
    {
        const int num_allocs = 1000;
        const int iters = 1000;

        uvm_eviction_policy_t *policy = uvm_eviction_policy_init(EVICTION_POLICY_FIFO, 0);
        uvm_alloc_metadata_t **allocs = (uvm_alloc_metadata_t**)calloc(num_allocs, sizeof(uvm_alloc_metadata_t*));

        // Create and register allocations
        for (int i = 0; i < num_allocs; i++) {
            allocs[i] = create_test_allocation(4 * 1024 * 1024);
            uvm_eviction_policy_register(policy, allocs[i]);
        }

        // Benchmark victim selection
        uint64_t start = get_time_ns();
        for (int i = 0; i < iters; i++) {
            uvm_eviction_policy_select_victim(policy);
        }
        uint64_t end = get_time_ns();

        double avg_time_ns = (end - start) / (double)iters;

        printf("%-45s %.2f ns/selection\n", "FIFO Victim Selection (1000 allocs)", avg_time_ns);

        for (int i = 0; i < num_allocs; i++) {
            uvm_eviction_policy_unregister(policy, allocs[i]);
            free_test_allocation(allocs[i]);
        }
        free(allocs);
        uvm_eviction_policy_shutdown(policy);
    }

    // Benchmark 4: Scalability Test
    {
        const int sizes[] = {100, 500, 1000, 5000};
        const int iters = 1000;

        printf("%-45s ", "Victim Selection Scalability");
        for (int s = 0; s < 4; s++) {
            int num_allocs = sizes[s];

            uvm_eviction_policy_t *policy = uvm_eviction_policy_init(EVICTION_POLICY_ACCESS_AWARE, 0);
            uvm_alloc_metadata_t **allocs = (uvm_alloc_metadata_t**)calloc(num_allocs, sizeof(uvm_alloc_metadata_t*));

            for (int i = 0; i < num_allocs; i++) {
                allocs[i] = create_test_allocation(1 * 1024 * 1024);
                uvm_eviction_policy_register(policy, allocs[i]);
            }

            uint64_t start = get_time_ns();
            for (int i = 0; i < iters; i++) {
                uvm_eviction_policy_select_victim(policy);
            }
            uint64_t end = get_time_ns();

            double avg_time_us = (end - start) / (1000.0 * iters);
            printf("%.2fus@%d ", avg_time_us, num_allocs);

            for (int i = 0; i < num_allocs; i++) {
                uvm_eviction_policy_unregister(policy, allocs[i]);
                free_test_allocation(allocs[i]);
            }
            free(allocs);
            uvm_eviction_policy_shutdown(policy);
        }
        printf("\n");
    }

    // Benchmark 5: Policy Switching Overhead
    {
        const int iters = 10000;

        uvm_eviction_policy_t *policy = uvm_eviction_policy_init(EVICTION_POLICY_LRU, 0);

        uint64_t start = get_time_ns();
        for (int i = 0; i < iters; i++) {
            uvm_eviction_policy_switch(policy, EVICTION_POLICY_ACCESS_AWARE);
            uvm_eviction_policy_switch(policy, EVICTION_POLICY_LRU);
        }
        uint64_t end = get_time_ns();

        double avg_time_ns = (end - start) / (double)(iters * 2);

        printf("%-45s %.2f ns/switch\n", "Policy Switching Overhead", avg_time_ns);

        uvm_eviction_policy_shutdown(policy);
    }

    // Benchmark 6: Access Update Overhead
    {
        const int num_allocs = 100;
        const int iters = 10000;

        uvm_eviction_policy_t *policy = uvm_eviction_policy_init(EVICTION_POLICY_ACCESS_AWARE, 0);
        uvm_alloc_metadata_t **allocs = (uvm_alloc_metadata_t**)calloc(num_allocs, sizeof(uvm_alloc_metadata_t*));

        for (int i = 0; i < num_allocs; i++) {
            allocs[i] = create_test_allocation(4 * 1024 * 1024);
            uvm_eviction_policy_register(policy, allocs[i]);
        }

        uint64_t start = get_time_ns();
        for (int i = 0; i < iters; i++) {
            uvm_eviction_policy_update_access(policy, allocs[i % num_allocs]);
        }
        uint64_t end = get_time_ns();

        double avg_time_ns = (end - start) / (double)iters;

        printf("%-45s %.2f ns/update\n", "Access Tracking Update Overhead", avg_time_ns);

        for (int i = 0; i < num_allocs; i++) {
            uvm_eviction_policy_unregister(policy, allocs[i]);
            free_test_allocation(allocs[i]);
        }
        free(allocs);
        uvm_eviction_policy_shutdown(policy);
    }

    // Benchmark 7: Thrashing Detection Overhead
    {
        const int iters = 10000;

        uvm_eviction_policy_t *policy = uvm_eviction_policy_init(EVICTION_POLICY_ACCESS_AWARE, 0);

        uint64_t start = get_time_ns();
        for (int i = 0; i < iters; i++) {
            uvm_eviction_policy_record_eviction(policy);
        }
        uint64_t end = get_time_ns();

        double avg_time_ns = (end - start) / (double)iters;

        printf("%-45s %.2f ns/record\n", "Eviction Recording + Thrashing Detection", avg_time_ns);

        uvm_eviction_policy_shutdown(policy);
    }

    printf("\n");
}
