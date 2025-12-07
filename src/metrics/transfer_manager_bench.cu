/**
 * Transfer Manager Benchmarks
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include "../../../bud_fcsp/src/uvm/transfer_manager.h"
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
    return alloc;
}

static void free_test_allocation(uvm_alloc_metadata_t *alloc) {
    if (!alloc) return;
    if (alloc->gpu_ptr) cudaFree(alloc->gpu_ptr);
    if (alloc->cpu_ptr) free(alloc->cpu_ptr);
    free(alloc);
}

extern "C" void run_transfer_manager_benchmarks(void) {
    printf("\n=================================================\n");
    printf("Transfer Manager Benchmarks\n");
    printf("=================================================\n\n");

    // Benchmark 1: Pinned Transfer Bandwidth
    {
        const size_t size = 256 * 1024 * 1024;
        const int iters = 5;
        
        uvm_transfer_manager_t *mgr = uvm_transfer_manager_init(0, TRANSFER_OPT_PINNED_MEMORY);
        uvm_alloc_metadata_t *alloc = create_test_allocation(size);
        
        uint64_t start = get_time_ns();
        for (int i = 0; i < iters; i++) {
            uvm_transfer_evict(mgr, alloc);
            cudaDeviceSynchronize();
        }
        uint64_t end = get_time_ns();
        
        double time_s = (end - start) / 1e9;
        double bw = (size * iters / (1024.0*1024.0*1024.0)) / time_s;
        
        printf("%-45s %.2f GB/s\n", "Pinned Memory Transfer Bandwidth", bw);
        
        free_test_allocation(alloc);
        uvm_transfer_manager_shutdown(mgr);
    }

    // Benchmark 2: Copy Engine Count
    {
        uvm_copy_engine_info_t info;
        uvm_transfer_query_copy_engines(0, &info);
        printf("%-45s %d engines\n", "Copy Engine Count", info.num_copy_engines);
        if (info.h2d_stream) cudaStreamDestroy(info.h2d_stream);
        if (info.d2h_stream) cudaStreamDestroy(info.d2h_stream);
    }

    // Benchmark 3: Batch Transfer
    {
        const int batch_size = 16;
        const size_t alloc_size = 4 * 1024 * 1024;
        
        uvm_transfer_manager_t *mgr = uvm_transfer_manager_init(0, TRANSFER_OPT_PINNED_MEMORY);
        uvm_alloc_metadata_t **allocs = (uvm_alloc_metadata_t**)calloc(batch_size, sizeof(uvm_alloc_metadata_t*));
        for (int i = 0; i < batch_size; i++) {
            allocs[i] = create_test_allocation(alloc_size);
        }
        
        uint64_t start = get_time_ns();
        uvm_transfer_batch(mgr, allocs, batch_size, TRANSFER_GPU_TO_CPU);
        cudaDeviceSynchronize();
        uint64_t end = get_time_ns();
        
        double time_s = (end - start) / 1e9;
        double bw = (alloc_size * batch_size / (1024.0*1024.0*1024.0)) / time_s;
        
        printf("%-45s %.2f GB/s\n", "Batch Transfer Throughput", bw);
        
        for (int i = 0; i < batch_size; i++) free_test_allocation(allocs[i]);
        free(allocs);
        uvm_transfer_manager_shutdown(mgr);
    }

    printf("\n");
}
