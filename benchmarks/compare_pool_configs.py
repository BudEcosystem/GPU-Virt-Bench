#!/usr/bin/env python3
"""
Pool Configuration Benchmark Analysis
Compares different floor/shared/burst configurations
"""

import json
import os
from typing import Dict, List

def load_results(filepath: str) -> Dict:
    with open(filepath, 'r') as f:
        return json.load(f)

def get_metric_by_id(metrics: List[Dict], metric_id: str) -> Dict:
    for m in metrics:
        if m['id'] == metric_id:
            return m
    return None

def main():
    benchmark_dir = os.path.dirname(os.path.abspath(__file__))

    # Pool configurations to compare
    configs = [
        ('default (20/40/40)', 'fcsp_adaptive_isolation_results.json'),
        ('high_floor (40/30/30)', 'fcsp_high_floor_results.json'),
        ('very_high_floor (60/20/20)', 'fcsp_very_high_floor_results.json'),
        ('high_shared (10/60/30)', 'fcsp_high_shared_results.json'),
        ('high_burst (10/30/60)', 'fcsp_high_burst_results.json'),
        ('balanced_low (10/45/45)', 'fcsp_balanced_low_results.json'),
        ('no_burst (50/50/0)', 'fcsp_no_burst_results.json'),
        ('HAMi baseline', 'hami_50mem_75sm_results.json'),
    ]

    # Key metrics to analyze
    metrics_to_check = [
        ('IS-009', 'Noisy Neighbor Impact', '%', 'lower'),
        ('IS-006', 'Cross-Tenant Isolation', '%', 'higher'),
        ('IS-007', 'QoS Consistency (CV)', 'ratio', 'lower'),
        ('IS-008', 'Fairness Index', 'ratio', 'higher'),
        ('OH-001', 'Kernel Launch Latency', 'us', 'lower'),
        ('OH-008', 'Rate Limiter Overhead', 'ns', 'lower'),
        ('OH-010', 'Throughput Degradation', '%', 'lower'),
    ]

    print("=" * 120)
    print("FCSP POOL CONFIGURATION BENCHMARK ANALYSIS")
    print("=" * 120)
    print("\nConfiguration Format: (floor% / shared% / burst%)")
    print("-" * 120)

    # Load all results
    results = {}
    for name, filename in configs:
        filepath = os.path.join(benchmark_dir, filename)
        if os.path.exists(filepath):
            results[name] = load_results(filepath)
            print(f"Loaded: {name}")

    print("\n" + "=" * 120)
    print("RESULTS COMPARISON")
    print("=" * 120)

    # Print header
    header = f"{'Metric':<30}"
    for name, _ in configs:
        short_name = name.split(' ')[0][:15]
        header += f"{short_name:>14}"
    print(header)
    print("-" * 120)

    # Collect data for analysis
    metric_data = {}

    for metric_id, metric_name, unit, better in metrics_to_check:
        row = f"{metric_name:<30}"
        values = []

        for name, _ in configs:
            if name in results:
                metric = get_metric_by_id(results[name]['metrics'], metric_id)
                if metric:
                    val = metric['stats']['mean']
                    values.append((name, val))
                    if unit == '%':
                        row += f"{val:>13.2f}%"
                    elif unit == 'ratio':
                        row += f"{val:>14.4f}"
                    elif unit == 'us':
                        row += f"{val:>12.2f}us"
                    elif unit == 'ns':
                        row += f"{val:>12.0f}ns"
                else:
                    row += f"{'N/A':>14}"
            else:
                row += f"{'N/A':>14}"

        metric_data[metric_id] = (metric_name, values, better)
        print(row)

    print("-" * 120)

    # Analysis summary
    print("\n" + "=" * 120)
    print("KEY FINDINGS")
    print("=" * 120)

    # Find best config for noisy neighbor
    nn_data = metric_data.get('IS-009')
    if nn_data:
        _, values, _ = nn_data
        sorted_vals = sorted(values, key=lambda x: x[1])
        print(f"\n1. NOISY NEIGHBOR PROTECTION (lower is better):")
        for i, (name, val) in enumerate(sorted_vals[:5]):
            marker = " <-- BEST" if i == 0 else ""
            print(f"   {i+1}. {name}: {val:.2f}%{marker}")

    # Find best config for cross-tenant isolation
    ct_data = metric_data.get('IS-006')
    if ct_data:
        _, values, _ = ct_data
        sorted_vals = sorted(values, key=lambda x: x[1], reverse=True)
        print(f"\n2. CROSS-TENANT COMPUTE ISOLATION (higher is better):")
        for i, (name, val) in enumerate(sorted_vals[:5]):
            marker = " <-- BEST" if i == 0 else ""
            print(f"   {i+1}. {name}: {val:.2f}%{marker}")

    # QoS consistency
    qos_data = metric_data.get('IS-007')
    if qos_data:
        _, values, _ = qos_data
        sorted_vals = sorted(values, key=lambda x: x[1])
        print(f"\n3. QoS CONSISTENCY (lower CV is better):")
        for i, (name, val) in enumerate(sorted_vals[:5]):
            marker = " <-- BEST" if i == 0 else ""
            print(f"   {i+1}. {name}: {val:.4f}{marker}")

    # Overhead
    oh_data = metric_data.get('OH-010')
    if oh_data:
        _, values, _ = oh_data
        sorted_vals = sorted(values, key=lambda x: x[1])
        print(f"\n4. THROUGHPUT DEGRADATION (lower is better):")
        for i, (name, val) in enumerate(sorted_vals[:5]):
            marker = " <-- BEST" if i == 0 else ""
            print(f"   {i+1}. {name}: {val:.3f}%{marker}")

    print("\n" + "=" * 120)
    print("RECOMMENDATIONS")
    print("=" * 120)

    print("""
    Based on the benchmark results:

    1. FOR MAXIMUM NOISY NEIGHBOR PROTECTION:
       - Use HIGH FLOOR configuration (40-60% floor)
       - Recommended: BUD_TENANT_FLOOR_PCT=40-60, BUD_SHARED_POOL_PCT=20-30, BUD_BURST_MAX_PCT=20-30
       - Trade-off: Slightly lower resource utilization efficiency

    2. FOR MAXIMUM THROUGHPUT:
       - Use DEFAULT or HIGH SHARED configuration
       - Recommended: BUD_TENANT_FLOOR_PCT=10-20, BUD_SHARED_POOL_PCT=40-60, BUD_BURST_MAX_PCT=30-40
       - Trade-off: Higher noisy neighbor impact

    3. FOR BALANCED MULTI-TENANT WORKLOADS:
       - Use MODERATE FLOOR with ADAPTIVE isolation
       - Recommended: BUD_TENANT_FLOOR_PCT=30-40, BUD_SHARED_POOL_PCT=30-40, BUD_BURST_MAX_PCT=20-30
       - Best balance between isolation and efficiency

    4. AVOID:
       - Very low floor (<10%) in noisy neighbor sensitive environments
       - Zero burst pool if you have bursty LLM workloads
       - Very high burst (>50%) without adequate floor protection
    """)

    print("=" * 120)

if __name__ == '__main__':
    main()
