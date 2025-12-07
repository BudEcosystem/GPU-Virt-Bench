/*
 * GPU Virtualization Performance Evaluation Tool
 * Metric definitions and registry
 */

#ifndef GPU_VIRT_BENCH_METRICS_H
#define GPU_VIRT_BENCH_METRICS_H

#include "benchmark.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * ============================================================================
 * Metric Registry
 * All metrics are defined here with their properties
 * ============================================================================
 */

/* Total number of defined metrics (30 original + 26 new + 6 paper + 12 fcsp + 10 uvm = 84) */
#define TOTAL_METRIC_COUNT 84

/* Overhead metrics: OH-001 to OH-010 */
#define METRIC_OH_KERNEL_LAUNCH_LATENCY     "OH-001"
#define METRIC_OH_MEMORY_ALLOC_LATENCY      "OH-002"
#define METRIC_OH_MEMORY_FREE_LATENCY       "OH-003"
#define METRIC_OH_CONTEXT_CREATION          "OH-004"
#define METRIC_OH_API_INTERCEPTION          "OH-005"
#define METRIC_OH_LOCK_CONTENTION           "OH-006"
#define METRIC_OH_MEMORY_TRACKING           "OH-007"
#define METRIC_OH_RATE_LIMITER              "OH-008"
#define METRIC_OH_NVML_POLLING              "OH-009"
#define METRIC_OH_TOTAL_THROUGHPUT          "OH-010"

/* Isolation metrics: IS-001 to IS-010 */
#define METRIC_IS_MEMORY_LIMIT_ACCURACY     "IS-001"
#define METRIC_IS_MEMORY_LIMIT_ENFORCEMENT  "IS-002"
#define METRIC_IS_SM_UTIL_ACCURACY          "IS-003"
#define METRIC_IS_SM_LIMIT_RESPONSE         "IS-004"
#define METRIC_IS_CROSS_TENANT_MEMORY       "IS-005"
#define METRIC_IS_CROSS_TENANT_COMPUTE      "IS-006"
#define METRIC_IS_QOS_CONSISTENCY           "IS-007"
#define METRIC_IS_FAIRNESS_INDEX            "IS-008"
#define METRIC_IS_NOISY_NEIGHBOR            "IS-009"
#define METRIC_IS_FAULT_ISOLATION           "IS-010"

/* LLM metrics: LLM-001 to LLM-010 */
#define METRIC_LLM_ATTENTION_THROUGHPUT     "LLM-001"
#define METRIC_LLM_KV_CACHE_ALLOC           "LLM-002"
#define METRIC_LLM_BATCH_SIZE_SCALING       "LLM-003"
#define METRIC_LLM_TOKEN_LATENCY            "LLM-004"
#define METRIC_LLM_MEMORY_POOL              "LLM-005"
#define METRIC_LLM_MULTI_STREAM             "LLM-006"
#define METRIC_LLM_LARGE_TENSOR             "LLM-007"
#define METRIC_LLM_MIXED_PRECISION          "LLM-008"
#define METRIC_LLM_DYNAMIC_BATCHING         "LLM-009"
#define METRIC_LLM_MULTI_GPU_SCALING        "LLM-010"

/* Memory Bandwidth metrics: BW-001 to BW-004 */
#define METRIC_BW_ISOLATION                 "BW-001"
#define METRIC_BW_FAIRNESS                  "BW-002"
#define METRIC_BW_SATURATION                "BW-003"
#define METRIC_BW_INTERFERENCE              "BW-004"

/* Cache Isolation metrics: CACHE-001 to CACHE-004 */
#define METRIC_CACHE_L2_HIT_RATE            "CACHE-001"
#define METRIC_CACHE_EVICTION_RATE          "CACHE-002"
#define METRIC_CACHE_WORKING_SET            "CACHE-003"
#define METRIC_CACHE_CONTENTION             "CACHE-004"

/* PCIe Bandwidth metrics: PCIE-001 to PCIE-004 */
#define METRIC_PCIE_H2D_BANDWIDTH           "PCIE-001"
#define METRIC_PCIE_D2H_BANDWIDTH           "PCIE-002"
#define METRIC_PCIE_CONTENTION              "PCIE-003"
#define METRIC_PCIE_PINNED_MEMORY           "PCIE-004"

/* NCCL/P2P Communication metrics: NCCL-001 to NCCL-004 */
#define METRIC_NCCL_ALLREDUCE_LATENCY       "NCCL-001"
#define METRIC_NCCL_ALLGATHER_BW            "NCCL-002"
#define METRIC_NCCL_P2P_BANDWIDTH           "NCCL-003"
#define METRIC_NCCL_BROADCAST_BW            "NCCL-004"

/* Aliases for compatibility */
#define METRIC_NCCL_ALLREDUCE               METRIC_NCCL_ALLREDUCE_LATENCY
#define METRIC_NCCL_ALLGATHER               METRIC_NCCL_ALLGATHER_BW
#define METRIC_NCCL_BROADCAST               METRIC_NCCL_BROADCAST_BW

/* Scheduling metrics: SCHED-001 to SCHED-004 */
#define METRIC_SCHED_CONTEXT_SWITCH         "SCHED-001"
#define METRIC_SCHED_KERNEL_LAUNCH          "SCHED-002"
#define METRIC_SCHED_STREAM_CONCURRENCY     "SCHED-003"
#define METRIC_SCHED_PREEMPTION             "SCHED-004"

/* Aliases for compatibility */
#define METRIC_SCHED_STREAM_PRIORITY        METRIC_SCHED_KERNEL_LAUNCH
#define METRIC_SCHED_WARP_FAIRNESS          METRIC_SCHED_STREAM_CONCURRENCY
#define METRIC_SCHED_HOL_BLOCKING           METRIC_SCHED_PREEMPTION

/* Memory Fragmentation metrics: FRAG-001 to FRAG-003 */
#define METRIC_FRAG_INDEX                   "FRAG-001"
#define METRIC_FRAG_ALLOC_LATENCY           "FRAG-002"
#define METRIC_FRAG_COMPACTION              "FRAG-003"

/* Aliases for compatibility */
#define METRIC_FRAG_EXTERNAL                METRIC_FRAG_INDEX
#define METRIC_FRAG_LARGEST_BLOCK           METRIC_FRAG_ALLOC_LATENCY
#define METRIC_FRAG_ALLOC_FAILURE           METRIC_FRAG_COMPACTION

/* Error Recovery metrics: ERR-001 to ERR-003 */
#define METRIC_ERR_DETECTION                "ERR-001"
#define METRIC_ERR_RECOVERY_TIME            "ERR-002"
#define METRIC_ERR_GRACEFUL_DEGRADATION     "ERR-003"

/* Aliases for compatibility */
#define METRIC_ERR_CASCADING                METRIC_ERR_RECOVERY_TIME
#define METRIC_ERR_RESET_REQUIRED           METRIC_ERR_GRACEFUL_DEGRADATION

/* Paper-inspired feature metrics: PAPER-001 to PAPER-006 */
#define METRIC_PAPER_GRAPH_COST_ACCURACY     "PAPER-001"
#define METRIC_PAPER_COMPUTE_CLASSIFICATION  "PAPER-002"
#define METRIC_PAPER_IO_CLASSIFICATION       "PAPER-003"
#define METRIC_PAPER_GRAPH_COST_PROPORTION   "PAPER-004"
#define METRIC_PAPER_WORKLOAD_FAIRNESS       "PAPER-005"
#define METRIC_PAPER_EXEC_MODEL_SPEEDUP      "PAPER-006"

/* FCSP Advanced Feature metrics: FCSP-001 to FCSP-012 */
#define METRIC_FCSP_AFFINITY_COMPLEMENTARY   "FCSP-001"
#define METRIC_FCSP_AFFINITY_CONFLICTING     "FCSP-002"
#define METRIC_FCSP_ADAPTIVE_SM_RESPONSE     "FCSP-003"
#define METRIC_FCSP_ADAPTIVE_SM_EFFICIENCY   "FCSP-004"
#define METRIC_FCSP_UVM_ALLOC_OVERHEAD       "FCSP-005"
#define METRIC_FCSP_UVM_PREFETCH_EFFECT      "FCSP-006"
#define METRIC_FCSP_UVM_TRANSFER_OVERLAP     "FCSP-007"
#define METRIC_FCSP_UVM_MEMORY_PRESSURE      "FCSP-008"
#define METRIC_FCSP_STREAM_ACTIVATION        "FCSP-009"
#define METRIC_FCSP_ATTENTION_PATTERN        "FCSP-010"
#define METRIC_FCSP_FFN_PATTERN              "FCSP-011"
#define METRIC_FCSP_MIXED_WORKLOAD           "FCSP-012"

/* UVM CPU Offloading metrics: UVM-001 to UVM-010 */
#define METRIC_UVM_PAGE_FAULT_RATE           "UVM-001"
#define METRIC_UVM_PREFETCH_HIT_RATE         "UVM-002"
#define METRIC_UVM_TRANSFER_HIDING           "UVM-003"
#define METRIC_UVM_CPU_MEM_OVERHEAD          "UVM-004"
#define METRIC_UVM_EVICTION_LATENCY          "UVM-005"
#define METRIC_UVM_PREFETCH_LATENCY          "UVM-006"
#define METRIC_UVM_OOM_PREVENTION            "UVM-007"
#define METRIC_UVM_BANDWIDTH_UTIL            "UVM-008"
#define METRIC_UVM_THRASHING                 "UVM-009"
#define METRIC_UVM_PRESSURE_RESPONSE         "UVM-010"

/*
 * ============================================================================
 * Extended Metric Categories (uses metric_category_t from benchmark.h)
 * Note: Extended categories are mapped to the 3 base categories for compatibility
 * ============================================================================
 */

/* New categories are mapped to base categories for compatibility */
#define METRIC_CATEGORY_BANDWIDTH      METRIC_CATEGORY_ISOLATION
#define METRIC_CATEGORY_CACHE          METRIC_CATEGORY_ISOLATION
#define METRIC_CATEGORY_PCIE           METRIC_CATEGORY_OVERHEAD
#define METRIC_CATEGORY_NCCL           METRIC_CATEGORY_LLM
#define METRIC_CATEGORY_SCHEDULING     METRIC_CATEGORY_OVERHEAD
#define METRIC_CATEGORY_FRAGMENTATION  METRIC_CATEGORY_OVERHEAD
#define METRIC_CATEGORY_ERROR          METRIC_CATEGORY_ISOLATION

/*
 * ============================================================================
 * Metric Definition Table
 * ============================================================================
 */

static const metric_definition_t METRIC_DEFINITIONS[TOTAL_METRIC_COUNT] = {
    /* ========== OVERHEAD METRICS (OH-001 to OH-010) ========== */
    {
        .id = "OH-001",
        .name = "Kernel Launch Latency",
        .description = "Time from cuLaunchKernel API call to kernel execution start on GPU.",
        .category = METRIC_CATEGORY_OVERHEAD,
        .unit = METRIC_UNIT_MICROSECONDS,
        .higher_is_better = false,
    },
    {
        .id = "OH-002",
        .name = "Memory Allocation Latency",
        .description = "Time for cuMemAlloc to complete including OOM check and tracking.",
        .category = METRIC_CATEGORY_OVERHEAD,
        .unit = METRIC_UNIT_MICROSECONDS,
        .higher_is_better = false,
    },
    {
        .id = "OH-003",
        .name = "Memory Free Latency",
        .description = "Time for cuMemFree to complete including memory tracking cleanup.",
        .category = METRIC_CATEGORY_OVERHEAD,
        .unit = METRIC_UNIT_MICROSECONDS,
        .higher_is_better = false,
    },
    {
        .id = "OH-004",
        .name = "Context Creation Overhead",
        .description = "Additional time for CUDA context creation compared to native.",
        .category = METRIC_CATEGORY_OVERHEAD,
        .unit = METRIC_UNIT_MICROSECONDS,
        .higher_is_better = false,
    },
    {
        .id = "OH-005",
        .name = "API Interception Overhead",
        .description = "Per-call overhead of dlsym hooking and function dispatch.",
        .category = METRIC_CATEGORY_OVERHEAD,
        .unit = METRIC_UNIT_NANOSECONDS,
        .higher_is_better = false,
    },
    {
        .id = "OH-006",
        .name = "Shared Region Lock Contention",
        .description = "Time spent waiting for shared region lock under contention.",
        .category = METRIC_CATEGORY_OVERHEAD,
        .unit = METRIC_UNIT_MICROSECONDS,
        .higher_is_better = false,
    },
    {
        .id = "OH-007",
        .name = "Memory Tracking Overhead",
        .description = "Per-allocation cost of memory accounting excluding lock wait.",
        .category = METRIC_CATEGORY_OVERHEAD,
        .unit = METRIC_UNIT_NANOSECONDS,
        .higher_is_better = false,
    },
    {
        .id = "OH-008",
        .name = "Rate Limiter Overhead",
        .description = "Time spent in rate limiter function including token bucket check.",
        .category = METRIC_CATEGORY_OVERHEAD,
        .unit = METRIC_UNIT_NANOSECONDS,
        .higher_is_better = false,
    },
    {
        .id = "OH-009",
        .name = "NVML Polling Overhead",
        .description = "CPU utilization consumed by NVML polling thread.",
        .category = METRIC_CATEGORY_OVERHEAD,
        .unit = METRIC_UNIT_PERCENTAGE,
        .higher_is_better = false,
    },
    {
        .id = "OH-010",
        .name = "Total Throughput Degradation",
        .description = "End-to-end throughput reduction compared to native execution.",
        .category = METRIC_CATEGORY_OVERHEAD,
        .unit = METRIC_UNIT_PERCENTAGE,
        .higher_is_better = false,
    },

    /* ========== ISOLATION METRICS (IS-001 to IS-010) ========== */
    {
        .id = "IS-001",
        .name = "Memory Limit Accuracy",
        .description = "Ratio of actual enforced memory limit to configured limit.",
        .category = METRIC_CATEGORY_ISOLATION,
        .unit = METRIC_UNIT_PERCENTAGE,
        .higher_is_better = true,
    },
    {
        .id = "IS-002",
        .name = "Memory Limit Enforcement Time",
        .description = "Time from allocation request exceeding limit to OOM error return.",
        .category = METRIC_CATEGORY_ISOLATION,
        .unit = METRIC_UNIT_MICROSECONDS,
        .higher_is_better = false,
    },
    {
        .id = "IS-003",
        .name = "SM Utilization Accuracy",
        .description = "Ratio of actual SM utilization to configured SM limit.",
        .category = METRIC_CATEGORY_ISOLATION,
        .unit = METRIC_UNIT_PERCENTAGE,
        .higher_is_better = true,
    },
    {
        .id = "IS-004",
        .name = "SM Limit Response Time",
        .description = "Time for SM utilization to converge to target after workload change.",
        .category = METRIC_CATEGORY_ISOLATION,
        .unit = METRIC_UNIT_MILLISECONDS,
        .higher_is_better = false,
    },
    {
        .id = "IS-005",
        .name = "Cross-Tenant Memory Isolation",
        .description = "Binary test: can one tenant access another tenant's GPU memory?",
        .category = METRIC_CATEGORY_ISOLATION,
        .unit = METRIC_UNIT_BOOLEAN,
        .higher_is_better = true,
    },
    {
        .id = "IS-006",
        .name = "Cross-Tenant Compute Isolation",
        .description = "Percentage of throughput maintained when other tenants are active.",
        .category = METRIC_CATEGORY_ISOLATION,
        .unit = METRIC_UNIT_PERCENTAGE,
        .higher_is_better = true,
    },
    {
        .id = "IS-007",
        .name = "QoS Consistency",
        .description = "Coefficient of variation of performance over time.",
        .category = METRIC_CATEGORY_ISOLATION,
        .unit = METRIC_UNIT_RATIO,
        .higher_is_better = false,
    },
    {
        .id = "IS-008",
        .name = "Fairness Index",
        .description = "Jain's fairness index across tenants with equal allocation.",
        .category = METRIC_CATEGORY_ISOLATION,
        .unit = METRIC_UNIT_RATIO,
        .higher_is_better = true,
    },
    {
        .id = "IS-009",
        .name = "Noisy Neighbor Impact",
        .description = "Performance degradation from aggressive neighbor workload.",
        .category = METRIC_CATEGORY_ISOLATION,
        .unit = METRIC_UNIT_PERCENTAGE,
        .higher_is_better = false,
    },
    {
        .id = "IS-010",
        .name = "Fault Isolation",
        .description = "Binary test: does a fault in one tenant affect others?",
        .category = METRIC_CATEGORY_ISOLATION,
        .unit = METRIC_UNIT_BOOLEAN,
        .higher_is_better = true,
    },

    /* ========== LLM METRICS (LLM-001 to LLM-010) ========== */
    {
        .id = "LLM-001",
        .name = "Attention Kernel Throughput",
        .description = "TFLOPS achieved on transformer self-attention computation.",
        .category = METRIC_CATEGORY_LLM,
        .unit = METRIC_UNIT_FLOPS,
        .higher_is_better = true,
    },
    {
        .id = "LLM-002",
        .name = "KV Cache Allocation Speed",
        .description = "Allocations per second for KV cache page-style allocations.",
        .category = METRIC_CATEGORY_LLM,
        .unit = METRIC_UNIT_ALLOCS_PER_SEC,
        .higher_is_better = true,
    },
    {
        .id = "LLM-003",
        .name = "Batch Size Scaling Efficiency",
        .description = "Throughput scaling factor as batch size increases.",
        .category = METRIC_CATEGORY_LLM,
        .unit = METRIC_UNIT_RATIO,
        .higher_is_better = true,
    },
    {
        .id = "LLM-004",
        .name = "Token Generation Latency",
        .description = "Time per output token in autoregressive generation.",
        .category = METRIC_CATEGORY_LLM,
        .unit = METRIC_UNIT_MILLISECONDS,
        .higher_is_better = false,
    },
    {
        .id = "LLM-005",
        .name = "Memory Pool Efficiency",
        .description = "Overhead when using CUDA memory pools vs direct allocation.",
        .category = METRIC_CATEGORY_LLM,
        .unit = METRIC_UNIT_PERCENTAGE,
        .higher_is_better = false,
    },
    {
        .id = "LLM-006",
        .name = "Multi-Stream Performance",
        .description = "Efficiency of concurrent stream execution.",
        .category = METRIC_CATEGORY_LLM,
        .unit = METRIC_UNIT_PERCENTAGE,
        .higher_is_better = true,
    },
    {
        .id = "LLM-007",
        .name = "Large Tensor Allocation Time",
        .description = "Time to allocate large contiguous tensors for model weights.",
        .category = METRIC_CATEGORY_LLM,
        .unit = METRIC_UNIT_MILLISECONDS,
        .higher_is_better = false,
    },
    {
        .id = "LLM-008",
        .name = "Mixed Precision Throughput Ratio",
        .description = "FP16/BF16 throughput relative to FP32 baseline.",
        .category = METRIC_CATEGORY_LLM,
        .unit = METRIC_UNIT_RATIO,
        .higher_is_better = true,
    },
    {
        .id = "LLM-009",
        .name = "Dynamic Batching Latency Variance",
        .description = "CV in latency under dynamic batch size changes.",
        .category = METRIC_CATEGORY_LLM,
        .unit = METRIC_UNIT_RATIO,
        .higher_is_better = false,
    },
    {
        .id = "LLM-010",
        .name = "Multi-GPU Scaling Factor",
        .description = "Throughput scaling factor across multiple GPUs.",
        .category = METRIC_CATEGORY_LLM,
        .unit = METRIC_UNIT_RATIO,
        .higher_is_better = true,
    },

    /* ========== MEMORY BANDWIDTH METRICS (BW-001 to BW-004) ========== */
    {
        .id = "BW-001",
        .name = "Memory Bandwidth Isolation",
        .description = "Percentage of theoretical bandwidth achieved under contention.",
        .category = METRIC_CATEGORY_ISOLATION,
        .unit = METRIC_UNIT_PERCENTAGE,
        .higher_is_better = true,
    },
    {
        .id = "BW-002",
        .name = "Bandwidth Fairness Index",
        .description = "Jain's fairness index for bandwidth distribution across tenants.",
        .category = METRIC_CATEGORY_ISOLATION,
        .unit = METRIC_UNIT_RATIO,
        .higher_is_better = true,
    },
    {
        .id = "BW-003",
        .name = "Memory Bus Saturation Point",
        .description = "Number of concurrent memory streams before saturation.",
        .category = METRIC_CATEGORY_ISOLATION,
        .unit = METRIC_UNIT_RATIO,
        .higher_is_better = true,
    },
    {
        .id = "BW-004",
        .name = "Bandwidth Interference Impact",
        .description = "Percentage bandwidth drop from competing memory workloads.",
        .category = METRIC_CATEGORY_ISOLATION,
        .unit = METRIC_UNIT_PERCENTAGE,
        .higher_is_better = false,
    },

    /* ========== CACHE ISOLATION METRICS (CACHE-001 to CACHE-004) ========== */
    {
        .id = "CACHE-001",
        .name = "L2 Cache Hit Rate",
        .description = "L2 cache hit rate under multi-tenant workloads.",
        .category = METRIC_CATEGORY_ISOLATION,
        .unit = METRIC_UNIT_PERCENTAGE,
        .higher_is_better = true,
    },
    {
        .id = "CACHE-002",
        .name = "Cache Eviction Rate",
        .description = "Rate of cache evictions caused by other tenants.",
        .category = METRIC_CATEGORY_ISOLATION,
        .unit = METRIC_UNIT_PERCENTAGE,
        .higher_is_better = false,
    },
    {
        .id = "CACHE-003",
        .name = "Working Set Collision Impact",
        .description = "Performance drop when working sets overlap in cache.",
        .category = METRIC_CATEGORY_ISOLATION,
        .unit = METRIC_UNIT_PERCENTAGE,
        .higher_is_better = false,
    },
    {
        .id = "CACHE-004",
        .name = "Cache Contention Overhead",
        .description = "Additional latency from L2 cache contention.",
        .category = METRIC_CATEGORY_ISOLATION,
        .unit = METRIC_UNIT_PERCENTAGE,
        .higher_is_better = false,
    },

    /* ========== PCIE BANDWIDTH METRICS (PCIE-001 to PCIE-004) ========== */
    {
        .id = "PCIE-001",
        .name = "Host-to-Device Bandwidth",
        .description = "Achieved H2D bandwidth as percentage of theoretical max.",
        .category = METRIC_CATEGORY_OVERHEAD,
        .unit = METRIC_UNIT_BANDWIDTH,
        .higher_is_better = true,
    },
    {
        .id = "PCIE-002",
        .name = "Device-to-Host Bandwidth",
        .description = "Achieved D2H bandwidth as percentage of theoretical max.",
        .category = METRIC_CATEGORY_OVERHEAD,
        .unit = METRIC_UNIT_BANDWIDTH,
        .higher_is_better = true,
    },
    {
        .id = "PCIE-003",
        .name = "PCIe Contention Impact",
        .description = "Bandwidth drop under multi-tenant PCIe traffic.",
        .category = METRIC_CATEGORY_ISOLATION,
        .unit = METRIC_UNIT_PERCENTAGE,
        .higher_is_better = false,
    },
    {
        .id = "PCIE-004",
        .name = "Pinned Memory Performance",
        .description = "Performance ratio of pinned vs pageable memory transfers.",
        .category = METRIC_CATEGORY_OVERHEAD,
        .unit = METRIC_UNIT_RATIO,
        .higher_is_better = true,
    },

    /* ========== NCCL/P2P METRICS (NCCL-001 to NCCL-004) ========== */
    {
        .id = "NCCL-001",
        .name = "AllReduce Latency",
        .description = "Time for allreduce collective operation.",
        .category = METRIC_CATEGORY_LLM,
        .unit = METRIC_UNIT_MICROSECONDS,
        .higher_is_better = false,
    },
    {
        .id = "NCCL-002",
        .name = "AllGather Bandwidth",
        .description = "Achieved bandwidth for allgather collective.",
        .category = METRIC_CATEGORY_LLM,
        .unit = METRIC_UNIT_BANDWIDTH,
        .higher_is_better = true,
    },
    {
        .id = "NCCL-003",
        .name = "P2P GPU Bandwidth",
        .description = "Direct GPU-to-GPU bandwidth (NVLink or PCIe).",
        .category = METRIC_CATEGORY_LLM,
        .unit = METRIC_UNIT_BANDWIDTH,
        .higher_is_better = true,
    },
    {
        .id = "NCCL-004",
        .name = "Broadcast Bandwidth",
        .description = "Achieved bandwidth for broadcast collective.",
        .category = METRIC_CATEGORY_LLM,
        .unit = METRIC_UNIT_BANDWIDTH,
        .higher_is_better = true,
    },

    /* ========== SCHEDULING METRICS (SCHED-001 to SCHED-004) ========== */
    {
        .id = "SCHED-001",
        .name = "Context Switch Latency",
        .description = "Time to switch between CUDA contexts.",
        .category = METRIC_CATEGORY_OVERHEAD,
        .unit = METRIC_UNIT_MICROSECONDS,
        .higher_is_better = false,
    },
    {
        .id = "SCHED-002",
        .name = "Kernel Launch Overhead",
        .description = "Overhead of launching minimal kernels.",
        .category = METRIC_CATEGORY_OVERHEAD,
        .unit = METRIC_UNIT_MICROSECONDS,
        .higher_is_better = false,
    },
    {
        .id = "SCHED-003",
        .name = "Stream Concurrency Efficiency",
        .description = "Efficiency of concurrent stream execution.",
        .category = METRIC_CATEGORY_ISOLATION,
        .unit = METRIC_UNIT_PERCENTAGE,
        .higher_is_better = true,
    },
    {
        .id = "SCHED-004",
        .name = "Preemption Latency",
        .description = "Latency when higher priority work preempts.",
        .category = METRIC_CATEGORY_ISOLATION,
        .unit = METRIC_UNIT_MILLISECONDS,
        .higher_is_better = false,
    },

    /* ========== FRAGMENTATION METRICS (FRAG-001 to FRAG-003) ========== */
    {
        .id = "FRAG-001",
        .name = "Fragmentation Index",
        .description = "Memory fragmentation level after alloc/free cycles.",
        .category = METRIC_CATEGORY_OVERHEAD,
        .unit = METRIC_UNIT_PERCENTAGE,
        .higher_is_better = false,
    },
    {
        .id = "FRAG-002",
        .name = "Allocation Latency Degradation",
        .description = "How much allocation latency increases with fragmentation.",
        .category = METRIC_CATEGORY_OVERHEAD,
        .unit = METRIC_UNIT_RATIO,
        .higher_is_better = false,
    },
    {
        .id = "FRAG-003",
        .name = "Memory Compaction Efficiency",
        .description = "How well memory is reclaimed after defragmentation.",
        .category = METRIC_CATEGORY_OVERHEAD,
        .unit = METRIC_UNIT_PERCENTAGE,
        .higher_is_better = true,
    },

    /* ========== ERROR RECOVERY METRICS (ERR-001 to ERR-003) ========== */
    {
        .id = "ERR-001",
        .name = "Error Detection Latency",
        .description = "Time to detect and report CUDA errors.",
        .category = METRIC_CATEGORY_ISOLATION,
        .unit = METRIC_UNIT_MICROSECONDS,
        .higher_is_better = false,
    },
    {
        .id = "ERR-002",
        .name = "Error Recovery Time",
        .description = "Time to recover GPU to usable state after error.",
        .category = METRIC_CATEGORY_ISOLATION,
        .unit = METRIC_UNIT_MICROSECONDS,
        .higher_is_better = false,
    },
    {
        .id = "ERR-003",
        .name = "Graceful Degradation Score",
        .description = "How well GPU handles resource exhaustion without crashing.",
        .category = METRIC_CATEGORY_ISOLATION,
        .unit = METRIC_UNIT_PERCENTAGE,
        .higher_is_better = true,
    },

    /* ========== PAPER-INSPIRED FEATURE METRICS (PAPER-001 to PAPER-006) ========== */
    {
        .id = "PAPER-001",
        .name = "Graph Cost Pre-computation Accuracy",
        .description = "Measures if pre-computed CUDA Graph costs reflect actual complexity. Based on paper insight of front-loading cost calculation.",
        .category = METRIC_CATEGORY_OVERHEAD,
        .unit = METRIC_UNIT_PERCENTAGE,
        .higher_is_better = true,
    },
    {
        .id = "PAPER-002",
        .name = "Compute-Intensive Classification",
        .description = "Multi-stream efficiency for compute-bound kernels. Should be ~100% (no benefit from concurrency).",
        .category = METRIC_CATEGORY_ISOLATION,
        .unit = METRIC_UNIT_PERCENTAGE,
        .higher_is_better = false,
    },
    {
        .id = "PAPER-003",
        .name = "IO-Intensive Classification",
        .description = "Multi-stream efficiency for IO-bound kernels. Should be >100% (benefits from concurrency per paper).",
        .category = METRIC_CATEGORY_ISOLATION,
        .unit = METRIC_UNIT_PERCENTAGE,
        .higher_is_better = true,
    },
    {
        .id = "PAPER-004",
        .name = "Graph Launch Cost Proportionality",
        .description = "Ratio of complex to simple graph launch times. Validates pre-computed cost tracking.",
        .category = METRIC_CATEGORY_OVERHEAD,
        .unit = METRIC_UNIT_RATIO,
        .higher_is_better = true,
    },
    {
        .id = "PAPER-005",
        .name = "Workload-Aware Throttling Fairness",
        .description = "Jain's fairness index for different workload types under adaptive throttling.",
        .category = METRIC_CATEGORY_ISOLATION,
        .unit = METRIC_UNIT_RATIO,
        .higher_is_better = true,
    },
    {
        .id = "PAPER-006",
        .name = "Execution Model Speedup",
        .description = "Speedup from concurrent vs sequential execution. Paper reports 2-6x for balanced workloads.",
        .category = METRIC_CATEGORY_LLM,
        .unit = METRIC_UNIT_RATIO,
        .higher_is_better = true,
    },

    /* ========== FCSP ADVANCED FEATURE METRICS (FCSP-001 to FCSP-012) ========== */
    {
        .id = "FCSP-001",
        .name = "Affinity Complementary Efficiency",
        .description = "Efficiency when running compute-heavy and memory-heavy workloads concurrently. >100% indicates benefit from affinity scheduling.",
        .category = METRIC_CATEGORY_ISOLATION,
        .unit = METRIC_UNIT_PERCENTAGE,
        .higher_is_better = true,
    },
    {
        .id = "FCSP-002",
        .name = "Affinity Conflict Detection",
        .description = "Slowdown ratio when running two compute-heavy workloads. ~2.0x indicates correct conflict detection and serialization.",
        .category = METRIC_CATEGORY_ISOLATION,
        .unit = METRIC_UNIT_RATIO,
        .higher_is_better = false,
    },
    {
        .id = "FCSP-003",
        .name = "Adaptive SM Response Time",
        .description = "Time for heavy workload after light workload with adaptive SM. Lower indicates faster adaptation to available resources.",
        .category = METRIC_CATEGORY_OVERHEAD,
        .unit = METRIC_UNIT_MILLISECONDS,
        .higher_is_better = false,
    },
    {
        .id = "FCSP-004",
        .name = "Adaptive SM Efficiency",
        .description = "Throughput achieved with bursty multi-stream workload. Higher indicates better utilization of available SM resources.",
        .category = METRIC_CATEGORY_LLM,
        .unit = METRIC_UNIT_BANDWIDTH,
        .higher_is_better = true,
    },
    {
        .id = "FCSP-005",
        .name = "UVM Allocation Overhead",
        .description = "Ratio of UVM allocation time vs native allocation. 1.0x = same as native, higher = more overhead.",
        .category = METRIC_CATEGORY_OVERHEAD,
        .unit = METRIC_UNIT_RATIO,
        .higher_is_better = false,
    },
    {
        .id = "FCSP-006",
        .name = "UVM Prefetch Effectiveness",
        .description = "Speedup from prefetching vs page faults. >1.0x indicates prefetch eliminates page fault latency.",
        .category = METRIC_CATEGORY_LLM,
        .unit = METRIC_UNIT_RATIO,
        .higher_is_better = true,
    },
    {
        .id = "FCSP-007",
        .name = "UVM Transfer Overlap Efficiency",
        .description = "Efficiency of overlapping data transfers with compute. >100% indicates transfers hidden behind compute.",
        .category = METRIC_CATEGORY_LLM,
        .unit = METRIC_UNIT_PERCENTAGE,
        .higher_is_better = true,
    },
    {
        .id = "FCSP-008",
        .name = "UVM Memory Pressure Handling",
        .description = "Execution time under GPU memory oversubscription. Tests graceful handling of memory pressure.",
        .category = METRIC_CATEGORY_ISOLATION,
        .unit = METRIC_UNIT_MILLISECONDS,
        .higher_is_better = false,
    },
    {
        .id = "FCSP-009",
        .name = "Stream Activation Latency",
        .description = "Time to reactivate an idle stream including prefetch. Lower indicates faster stream reactivation.",
        .category = METRIC_CATEGORY_OVERHEAD,
        .unit = METRIC_UNIT_MILLISECONDS,
        .higher_is_better = false,
    },
    {
        .id = "FCSP-010",
        .name = "Attention Pattern Throughput",
        .description = "TFLOPS achieved on transformer attention pattern workload. Tests small-grid large-block kernel efficiency.",
        .category = METRIC_CATEGORY_LLM,
        .unit = METRIC_UNIT_FLOPS,
        .higher_is_better = true,
    },
    {
        .id = "FCSP-011",
        .name = "FFN Pattern Throughput",
        .description = "TFLOPS achieved on transformer FFN pattern workload. Tests large-grid GEMM kernel efficiency.",
        .category = METRIC_CATEGORY_LLM,
        .unit = METRIC_UNIT_FLOPS,
        .higher_is_better = true,
    },
    {
        .id = "FCSP-012",
        .name = "Mixed Workload Orchestration",
        .description = "Operations per second with concurrent compute, memory, and balanced workloads. Tests overall system orchestration.",
        .category = METRIC_CATEGORY_LLM,
        .unit = METRIC_UNIT_THROUGHPUT,
        .higher_is_better = true,
    },

    /* ========== UVM CPU OFFLOADING METRICS (UVM-001 to UVM-010) ========== */
    {
        .id = "UVM-001",
        .name = "Page Fault Rate",
        .description = "Number of page faults per second when accessing evicted memory. Lower is better.",
        .category = METRIC_CATEGORY_OVERHEAD,
        .unit = METRIC_UNIT_THROUGHPUT,
        .higher_is_better = false,
    },
    {
        .id = "UVM-002",
        .name = "Prefetch Hit Rate",
        .description = "Percentage of prefetches that were used within 100ms. Higher indicates better prediction.",
        .category = METRIC_CATEGORY_LLM,
        .unit = METRIC_UNIT_PERCENTAGE,
        .higher_is_better = true,
    },
    {
        .id = "UVM-003",
        .name = "Transfer Latency Hiding",
        .description = "Percentage of transfer time overlapped with compute. >100% means fully hidden.",
        .category = METRIC_CATEGORY_LLM,
        .unit = METRIC_UNIT_PERCENTAGE,
        .higher_is_better = true,
    },
    {
        .id = "UVM-004",
        .name = "CPU Memory Overhead",
        .description = "Peak CPU memory used for offloading as percentage of GPU memory limit.",
        .category = METRIC_CATEGORY_OVERHEAD,
        .unit = METRIC_UNIT_PERCENTAGE,
        .higher_is_better = false,
    },
    {
        .id = "UVM-005",
        .name = "Eviction Latency",
        .description = "Time to evict 100MB from GPU to CPU memory.",
        .category = METRIC_CATEGORY_OVERHEAD,
        .unit = METRIC_UNIT_MILLISECONDS,
        .higher_is_better = false,
    },
    {
        .id = "UVM-006",
        .name = "Prefetch Latency",
        .description = "Time to prefetch 100MB from CPU to GPU memory.",
        .category = METRIC_CATEGORY_OVERHEAD,
        .unit = METRIC_UNIT_MILLISECONDS,
        .higher_is_better = false,
    },
    {
        .id = "UVM-007",
        .name = "OOM Prevention Success",
        .description = "Percentage of OOMs prevented through eviction. 100% means perfect.",
        .category = METRIC_CATEGORY_ISOLATION,
        .unit = METRIC_UNIT_PERCENTAGE,
        .higher_is_better = true,
    },
    {
        .id = "UVM-008",
        .name = "Bandwidth Utilization",
        .description = "Percentage of PCIe bandwidth used by UVM transfers.",
        .category = METRIC_CATEGORY_OVERHEAD,
        .unit = METRIC_UNIT_PERCENTAGE,
        .higher_is_better = false,
    },
    {
        .id = "UVM-009",
        .name = "Thrashing Detection",
        .description = "Percentage of allocations with repeated evict-prefetch cycles.",
        .category = METRIC_CATEGORY_ISOLATION,
        .unit = METRIC_UNIT_PERCENTAGE,
        .higher_is_better = false,
    },
    {
        .id = "UVM-010",
        .name = "Memory Pressure Response Time",
        .description = "Time from pressure threshold to eviction start.",
        .category = METRIC_CATEGORY_OVERHEAD,
        .unit = METRIC_UNIT_MILLISECONDS,
        .higher_is_better = false,
    },
};

/*
 * ============================================================================
 * Metric Registry Functions
 * ============================================================================
 */

const metric_definition_t* metrics_get_definition(const char *metric_id);
int metrics_get_by_category(metric_category_t category,
                           const metric_definition_t **definitions,
                           int max_count);
const char* metrics_get_unit_string(metric_unit_t unit);
bool metrics_is_valid_id(const char *metric_id);
int metrics_parse_list(const char *list_str, char metric_ids[][16], int max_count);
int metrics_format_value(const metric_definition_t *def, double value,
                        char *buffer, size_t buffer_size);
double metrics_calculate_composite_score(const metric_result_t *results, int count,
                                        const double *weights);

/*
 * ============================================================================
 * MIG Baseline Expected Values
 * ============================================================================
 */

typedef struct {
    char metric_id[16];
    double expected_value;
    double tolerance_percent;
    const char *notes;
} mig_expected_value_t;

static const mig_expected_value_t MIG_EXPECTED_VALUES[] = {
    /* Original metrics */
    {"OH-001", 5.0, 20.0, "MIG kernel launch ~5us"},
    {"OH-002", 50.0, 30.0, "MIG allocation similar to native"},
    {"OH-003", 20.0, 30.0, "MIG free similar to native"},
    {"OH-004", 100.0, 50.0, "MIG context creation slightly higher"},
    {"OH-005", 0.0, 0.0, "MIG has no API interception"},
    {"OH-006", 0.0, 0.0, "MIG has no shared region locks"},
    {"OH-007", 0.0, 0.0, "MIG has no software memory tracking"},
    {"OH-008", 0.0, 0.0, "MIG has no software rate limiting"},
    {"OH-009", 0.0, 0.0, "MIG has no NVML polling overhead"},
    {"OH-010", 2.0, 50.0, "MIG ~2% total overhead"},
    {"IS-001", 100.0, 0.0, "MIG perfect memory enforcement"},
    {"IS-002", 1.0, 50.0, "MIG instant OOM response"},
    {"IS-003", 100.0, 5.0, "MIG SM allocation is exact"},
    {"IS-004", 0.0, 0.0, "MIG SM assignment is static"},
    {"IS-005", 1.0, 0.0, "MIG hardware memory isolation"},
    {"IS-006", 100.0, 5.0, "MIG hardware compute isolation"},
    {"IS-007", 0.05, 50.0, "MIG QoS variance ~5%"},
    {"IS-008", 0.98, 2.0, "MIG fairness ~0.98"},
    {"IS-009", 2.0, 100.0, "MIG ~2% noisy neighbor"},
    {"IS-010", 1.0, 0.0, "MIG hardware fault isolation"},
    /* New metrics */
    {"BW-001", 95.0, 5.0, "MIG ~95% bandwidth isolation"},
    {"BW-002", 0.98, 2.0, "MIG bandwidth fairness ~0.98"},
    {"BW-003", 1.0, 10.0, "MIG dedicated memory controllers"},
    {"BW-004", 5.0, 50.0, "MIG ~5% bandwidth interference"},
    {"CACHE-001", 95.0, 5.0, "MIG partitioned L2 cache"},
    {"CACHE-002", 5.0, 50.0, "MIG ~5% eviction rate"},
    {"CACHE-003", 5.0, 50.0, "MIG ~5% working set collision"},
    {"CACHE-004", 5.0, 50.0, "MIG ~5% cache contention"},
    {"PCIE-001", 12.0, 10.0, "PCIe Gen4 ~12 GB/s H2D"},
    {"PCIE-002", 12.0, 10.0, "PCIe Gen4 ~12 GB/s D2H"},
    {"PCIE-003", 10.0, 50.0, "MIG ~10% PCIe contention"},
    {"PCIE-004", 1.5, 20.0, "Pinned ~1.5x faster"},
    {"SCHED-001", 5.0, 50.0, "MIG ~5us context switch"},
    {"SCHED-002", 1.0, 0.0, "MIG honors stream priority"},
    {"SCHED-003", 0.98, 2.0, "MIG warp fairness ~0.98"},
    {"SCHED-004", 0.0, 0.0, "MIG no HOL blocking"},
    {"FRAG-001", 5.0, 50.0, "MIG ~5% fragmentation"},
    {"FRAG-002", 95.0, 5.0, "MIG ~95% contiguous"},
    {"FRAG-003", 1.0, 100.0, "MIG ~1% alloc failures"},
    {"ERR-001", 10.0, 50.0, "MIG ~10ms error recovery"},
    {"ERR-002", 0.0, 0.0, "MIG no cascading failures"},
    {"ERR-003", 0.0, 0.0, "MIG no reset required"},
    /* Paper-inspired feature metrics */
    {"PAPER-001", 95.0, 10.0, "Graph cost accuracy ~95%"},
    {"PAPER-002", 100.0, 20.0, "Compute workload ~100% multi-stream efficiency"},
    {"PAPER-003", 200.0, 50.0, "IO workload ~200% multi-stream efficiency (paper: 4-6x possible)"},
    {"PAPER-004", 10.0, 100.0, "Complex/simple graph ratio ~10x"},
    {"PAPER-005", 0.95, 5.0, "Workload fairness ~0.95 Jain's index"},
    {"PAPER-006", 3.0, 50.0, "Execution model speedup ~3x (paper: 2-6x)"},
    /* FCSP advanced feature metrics */
    {"FCSP-001", 150.0, 30.0, "Affinity complementary ~150% efficiency (compute+memory overlap)"},
    {"FCSP-002", 2.0, 20.0, "Affinity conflict ~2.0x slowdown (correct serialization)"},
    {"FCSP-003", 10.0, 50.0, "Adaptive SM response ~10ms"},
    {"FCSP-004", 100.0, 30.0, "Adaptive SM efficiency ~100 GB/s"},
    {"FCSP-005", 1.5, 50.0, "UVM alloc overhead ~1.5x vs native"},
    {"FCSP-006", 5.0, 50.0, "UVM prefetch ~5x speedup vs page faults"},
    {"FCSP-007", 150.0, 30.0, "UVM transfer overlap ~150% efficiency"},
    {"FCSP-008", 50.0, 100.0, "UVM memory pressure ~50ms under oversubscription"},
    {"FCSP-009", 20.0, 50.0, "Stream activation ~20ms with prefetch"},
    {"FCSP-010", 1.0, 100.0, "Attention pattern ~1 TFLOPS (varies by GPU)"},
    {"FCSP-011", 1.0, 100.0, "FFN pattern ~1 TFLOPS (varies by GPU)"},
    {"FCSP-012", 100.0, 50.0, "Mixed workload ~100 ops/sec"},
    /* UVM CPU offloading metrics */
    {"UVM-001", 50.0, 100.0, "Page fault rate ~50 faults/sec (with UVM)"},
    {"UVM-002", 85.0, 20.0, "Prefetch hit rate ~85% (good prediction)"},
    {"UVM-003", 80.0, 30.0, "Transfer hiding ~80% overlap with compute"},
    {"UVM-004", 15.0, 50.0, "CPU memory overhead ~15% of GPU limit"},
    {"UVM-005", 3.0, 50.0, "Eviction latency ~3ms for 100MB (PCIe Gen4)"},
    {"UVM-006", 8.0, 50.0, "Prefetch latency ~8ms for 100MB"},
    {"UVM-007", 100.0, 5.0, "OOM prevention ~100% (perfect with UVM)"},
    {"UVM-008", 20.0, 50.0, "Bandwidth utilization ~20% of PCIe"},
    {"UVM-009", 3.0, 100.0, "Thrashing rate ~3% (minimal with good config)"},
    {"UVM-010", 0.5, 100.0, "Pressure response ~0.5ms (fast eviction trigger)"},
};

#define MIG_EXPECTED_VALUE_COUNT (sizeof(MIG_EXPECTED_VALUES) / sizeof(MIG_EXPECTED_VALUES[0]))

const mig_expected_value_t* metrics_get_mig_expected(const char *metric_id);
double metrics_mig_comparison_score(const char *metric_id, double actual_value);

#ifdef __cplusplus
}
#endif

#endif /* GPU_VIRT_BENCH_METRICS_H */
