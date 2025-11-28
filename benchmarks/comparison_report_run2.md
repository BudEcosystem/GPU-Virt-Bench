# GPU Virtualization Benchmark Comparison Report

Generated: 2025-11-29 02:40:48

## Systems Compared

- **hami**: 30 passed, 0 failed (100.0%)
- **fcsp**: 30 passed, 0 failed (100.0%)

## Summary

| Metric | hami | fcsp |
|--------|--------|--------|
| Pass Rate | 100.0% | 100.0% |
| Passed | 30 | 30 |
| Failed | 0 | 0 |

## Detailed Metrics

### IS

| Metric | Name | hami | fcsp |
|--------|------|------|------|
| IS-001 | Memory Limit Accuracy | 91.46 ✓ | 100.00 ✓ |
| IS-002 | Memory Limit Enforcement Time | 226.07 ✓ | 0.03 ✓ |
| IS-003 | SM Utilization Accuracy | 87.26 ✓ | 100.00 ✓ |
| IS-004 | SM Limit Response Time | 81.85 ✓ | 81.11 ✓ |
| IS-005 | Cross-Tenant Memory Isolation | 1.00 ✓ | 1.00 ✓ |
| IS-006 | Cross-Tenant Compute Isolation | 83.36 ✓ | 88.93 ✓ |
| IS-007 | QoS Consistency | 0.06 ✓ | 0.08 ✓ |
| IS-008 | Fairness Index | 1.00 ✓ | 1.00 ✓ |
| IS-009 | Noisy Neighbor Impact | 18.61 ✓ | 16.20 ✓ |
| IS-010 | Fault Isolation | 1.00 ✓ | 1.00 ✓ |

### LLM

| Metric | Name | hami | fcsp |
|--------|------|------|------|
| LLM-001 | Attention Kernel Throughput | 2396.68 ✓ | 2403.01 ✓ |
| LLM-002 | KV Cache Allocation Speed | 77503.07 ✓ | 73172.96 ✓ |
| LLM-003 | Batch Size Scaling Efficiency | 0.68 ✓ | 0.68 ✓ |
| LLM-004 | Token Generation Latency | 0.01 ✓ | 0.01 ✓ |
| LLM-005 | Memory Pool Efficiency | 3.08 ✓ | -4.16 ✓ |
| LLM-006 | Multi-Stream Performance | 79.57 ✓ | 68.46 ✓ |
| LLM-007 | Large Tensor Allocation Time | 1.97 ✓ | 1.97 ✓ |
| LLM-008 | Mixed Precision Throughput Ratio | 0.96 ✓ | 0.96 ✓ |
| LLM-009 | Dynamic Batching Latency Variance | 0.79 ✓ | 0.76 ✓ |
| LLM-010 | Multi-GPU Scaling Factor | 1.00 ✓ | 1.00 ✓ |

### OH

| Metric | Name | hami | fcsp |
|--------|------|------|------|
| OH-001 | Kernel Launch Latency | 4.36 ✓ | 4.42 ✓ |
| OH-002 | Memory Allocation Latency | 128.12 ✓ | 172.93 ✓ |
| OH-003 | Memory Free Latency | 437.13 ✓ | 437.64 ✓ |
| OH-004 | Context Creation Overhead | 82767.58 ✓ | 77.76 ✓ |
| OH-005 | API Interception Overhead | 38.00 ✓ | 18.00 ✓ |
| OH-006 | Shared Region Lock Contention | 65.66 ✓ | 177.78 ✓ |
| OH-007 | Memory Tracking Overhead | 0.00 ✓ | 0.00 ✓ |
| OH-008 | Rate Limiter Overhead | 1236.00 ✓ | 1319.00 ✓ |
| OH-009 | NVML Polling Overhead | 0.11 ✓ | 0.19 ✓ |
| OH-010 | Total Throughput Degradation | 0.00 ✓ | 0.00 ✓ |

