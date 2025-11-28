# GPU Virtualization Benchmark Comparison Report

Generated: 2025-11-29 01:53:16

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
| IS-002 | Memory Limit Enforcement Time | 202.12 ✓ | 0.03 ✓ |
| IS-003 | SM Utilization Accuracy | 99.20 ✓ | 100.00 ✓ |
| IS-004 | SM Limit Response Time | 80.29 ✓ | 85.41 ✓ |
| IS-005 | Cross-Tenant Memory Isolation | 1.00 ✓ | 1.00 ✓ |
| IS-006 | Cross-Tenant Compute Isolation | 82.80 ✓ | 78.03 ✓ |
| IS-007 | QoS Consistency | 0.08 ✓ | 0.07 ✓ |
| IS-008 | Fairness Index | 1.00 ✓ | 0.99 ✓ |
| IS-009 | Noisy Neighbor Impact | 16.14 ✓ | 4.45 ✓ |
| IS-010 | Fault Isolation | 1.00 ✓ | 1.00 ✓ |

### LLM

| Metric | Name | hami | fcsp |
|--------|------|------|------|
| LLM-001 | Attention Kernel Throughput | 2406.67 ✓ | 2391.16 ✓ |
| LLM-002 | KV Cache Allocation Speed | 83428.29 ✓ | 77309.05 ✓ |
| LLM-003 | Batch Size Scaling Efficiency | 0.67 ✓ | 0.68 ✓ |
| LLM-004 | Token Generation Latency | 0.01 ✓ | 0.01 ✓ |
| LLM-005 | Memory Pool Efficiency | -7.75 ✓ | -3.12 ✓ |
| LLM-006 | Multi-Stream Performance | 55.62 ✓ | 76.72 ✓ |
| LLM-007 | Large Tensor Allocation Time | 1.92 ✓ | 1.97 ✓ |
| LLM-008 | Mixed Precision Throughput Ratio | 0.97 ✓ | 0.97 ✓ |
| LLM-009 | Dynamic Batching Latency Variance | 0.80 ✓ | 0.76 ✓ |
| LLM-010 | Multi-GPU Scaling Factor | 1.00 ✓ | 1.00 ✓ |

### OH

| Metric | Name | hami | fcsp |
|--------|------|------|------|
| OH-001 | Kernel Launch Latency | 4.34 ✓ | 4.53 ✓ |
| OH-002 | Memory Allocation Latency | 91.39 ✓ | 90.53 ✓ |
| OH-003 | Memory Free Latency | 438.13 ✓ | 444.89 ✓ |
| OH-004 | Context Creation Overhead | 81934.11 ✓ | 77.55 ✓ |
| OH-005 | API Interception Overhead | 38.00 ✓ | 18.00 ✓ |
| OH-006 | Shared Region Lock Contention | 64.32 ✓ | 190.41 ✓ |
| OH-007 | Memory Tracking Overhead | 0.00 ✓ | 0.00 ✓ |
| OH-008 | Rate Limiter Overhead | 1251.00 ✓ | 1298.00 ✓ |
| OH-009 | NVML Polling Overhead | 0.10 ✓ | 0.44 ✓ |
| OH-010 | Total Throughput Degradation | 0.00 ✓ | 0.00 ✓ |

