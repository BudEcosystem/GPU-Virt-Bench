#!/usr/bin/env python3
"""
Benchmark Comparison Script
Compares FCSP vs HAMi vs Native baseline across all metrics
"""

import json
import os
from typing import Dict, List, Any

def load_results(filepath: str) -> Dict:
    """Load benchmark results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def get_metric_by_id(metrics: List[Dict], metric_id: str) -> Dict:
    """Get a metric by its ID"""
    for m in metrics:
        if m['id'] == metric_id:
            return m
    return None

def format_value(value: float, unit: str) -> str:
    """Format a value with appropriate precision"""
    if unit == 'ns':
        return f"{value:.2f} ns"
    elif unit == 'us':
        return f"{value:.2f} us"
    elif unit == 'ms':
        return f"{value:.2f} ms"
    elif unit == '%':
        return f"{value:.2f}%"
    elif unit == 'ratio':
        return f"{value:.4f}"
    elif unit == 'bool':
        return "PASS" if value >= 1 else "FAIL"
    elif unit == 'alloc/s':
        return f"{value:.0f}"
    elif unit == 'TFLOPS':
        return f"{value:.2f}"
    else:
        return f"{value:.4f}"

def compare_metrics(results_list: List[Dict], names: List[str]):
    """Compare metrics across multiple result sets"""

    # Define key metrics to compare
    key_metrics = [
        ("OH-001", "Kernel Launch Latency", "lower"),
        ("OH-002", "Memory Allocation Latency", "lower"),
        ("OH-003", "Memory Free Latency", "lower"),
        ("OH-005", "API Interception Overhead", "lower"),
        ("OH-008", "Rate Limiter Overhead", "lower"),
        ("OH-010", "Total Throughput Degradation", "lower"),
        ("IS-001", "Memory Limit Accuracy", "higher"),
        ("IS-006", "Cross-Tenant Compute Isolation", "higher"),
        ("IS-007", "QoS Consistency (CV)", "lower"),
        ("IS-008", "Fairness Index", "higher"),
        ("IS-009", "Noisy Neighbor Impact", "lower"),
        ("LLM-001", "Attention Kernel Throughput", "higher"),
        ("LLM-002", "KV Cache Allocation Speed", "higher"),
        ("LLM-004", "Token Generation Latency", "lower"),
        ("LLM-005", "Memory Pool Efficiency", "lower"),
    ]

    print("\n" + "="*100)
    print("FCSP vs HAMi COMPARATIVE BENCHMARK ANALYSIS")
    print("="*100)
    print(f"GPU: {results_list[0]['gpu']['name']}")
    print(f"GPU Memory: {results_list[0]['gpu']['total_memory_mb']} MB")
    print(f"SM Count: {results_list[0]['gpu']['sm_count']}")
    print("="*100 + "\n")

    # Header
    header = f"{'Metric':<45}"
    for name in names:
        header += f"{name:>18}"
    header += f"{'Winner':>12}"
    print(header)
    print("-"*100)

    fcsp_wins = 0
    hami_wins = 0
    ties = 0

    for metric_id, metric_name, better in key_metrics:
        row = f"{metric_name:<45}"
        values = []
        units = []

        for results in results_list:
            metric = get_metric_by_id(results['metrics'], metric_id)
            if metric:
                val = metric['stats']['mean'] if 'mean' in metric['stats'] else metric['value']
                values.append(val)
                units.append(metric['unit'])
                row += f"{format_value(val, metric['unit']):>18}"
            else:
                values.append(None)
                units.append('')
                row += f"{'N/A':>18}"

        # Determine winner (comparing FCSP vs HAMi, indices 1 and 2)
        winner = ""
        if len(values) >= 3 and values[1] is not None and values[2] is not None:
            fcsp_val = values[1]  # FCSP
            hami_val = values[2]  # HAMi

            if better == "lower":
                if fcsp_val < hami_val * 0.95:  # 5% margin
                    winner = "FCSP"
                    fcsp_wins += 1
                elif hami_val < fcsp_val * 0.95:
                    winner = "HAMi"
                    hami_wins += 1
                else:
                    winner = "TIE"
                    ties += 1
            else:  # higher is better
                if fcsp_val > hami_val * 1.05:
                    winner = "FCSP"
                    fcsp_wins += 1
                elif hami_val > fcsp_val * 1.05:
                    winner = "HAMi"
                    hami_wins += 1
                else:
                    winner = "TIE"
                    ties += 1

        row += f"{winner:>12}"
        print(row)

    print("-"*100)
    print(f"\nSUMMARY:")
    print(f"  FCSP Wins: {fcsp_wins}")
    print(f"  HAMi Wins: {hami_wins}")
    print(f"  Ties: {ties}")
    print()

    # Calculate overall scores
    print("\nDETAILED ANALYSIS BY CATEGORY:")
    print("="*100)

    categories = {
        "Overhead": ["OH-001", "OH-002", "OH-003", "OH-005", "OH-008", "OH-010"],
        "Isolation": ["IS-001", "IS-006", "IS-007", "IS-008", "IS-009"],
        "LLM Performance": ["LLM-001", "LLM-002", "LLM-004", "LLM-005"]
    }

    for category, metric_ids in categories.items():
        print(f"\n{category}:")
        print("-"*50)

        for metric_id in metric_ids:
            for results, name in zip(results_list, names):
                metric = get_metric_by_id(results['metrics'], metric_id)
                if metric:
                    mean = metric['stats'].get('mean', metric['value'])
                    std = metric['stats'].get('std_dev', 0)
                    print(f"  {name}: {metric['name'][:30]:<30} = {format_value(mean, metric['unit'])} (+/- {format_value(std, metric['unit'])})")

    # Key findings
    print("\n" + "="*100)
    print("KEY FINDINGS")
    print("="*100)

    # Find native baseline metrics
    native = results_list[0]
    fcsp = results_list[1]
    hami = results_list[2]

    # Overhead comparison
    native_oh1 = get_metric_by_id(native['metrics'], 'OH-001')['stats']['mean']
    fcsp_oh1 = get_metric_by_id(fcsp['metrics'], 'OH-001')['stats']['mean']
    hami_oh1 = get_metric_by_id(hami['metrics'], 'OH-001')['stats']['mean']

    print(f"\n1. KERNEL LAUNCH OVERHEAD:")
    print(f"   Native Baseline: {native_oh1:.2f} us")
    print(f"   FCSP Overhead: {((fcsp_oh1 - native_oh1) / native_oh1 * 100):.2f}% vs native")
    print(f"   HAMi Overhead: {((hami_oh1 - native_oh1) / native_oh1 * 100):.2f}% vs native")

    # Memory allocation
    native_oh2 = get_metric_by_id(native['metrics'], 'OH-002')['stats']['mean']
    fcsp_oh2 = get_metric_by_id(fcsp['metrics'], 'OH-002')['stats']['mean']
    hami_oh2 = get_metric_by_id(hami['metrics'], 'OH-002')['stats']['mean']

    print(f"\n2. MEMORY ALLOCATION LATENCY:")
    print(f"   Native Baseline: {native_oh2:.2f} us")
    print(f"   FCSP: {fcsp_oh2:.2f} us ({((fcsp_oh2 - native_oh2) / native_oh2 * 100):+.2f}%)")
    print(f"   HAMi: {hami_oh2:.2f} us ({((hami_oh2 - native_oh2) / native_oh2 * 100):+.2f}%)")

    # Isolation metrics
    fcsp_is6 = get_metric_by_id(fcsp['metrics'], 'IS-006')['stats']['mean']
    hami_is6 = get_metric_by_id(hami['metrics'], 'IS-006')['stats']['mean']

    print(f"\n3. CROSS-TENANT COMPUTE ISOLATION:")
    print(f"   FCSP: {fcsp_is6:.2f}% performance maintained under contention")
    print(f"   HAMi: {hami_is6:.2f}% performance maintained under contention")

    # Noisy neighbor
    fcsp_is9 = get_metric_by_id(fcsp['metrics'], 'IS-009')['stats']['mean']
    hami_is9 = get_metric_by_id(hami['metrics'], 'IS-009')['stats']['mean']

    print(f"\n4. NOISY NEIGHBOR PROTECTION:")
    print(f"   FCSP: {fcsp_is9:.2f}% degradation from noisy neighbors")
    print(f"   HAMi: {hami_is9:.2f}% degradation from noisy neighbors")

    # LLM performance
    fcsp_llm1 = get_metric_by_id(fcsp['metrics'], 'LLM-001')['stats']['mean']
    hami_llm1 = get_metric_by_id(hami['metrics'], 'LLM-001')['stats']['mean']

    print(f"\n5. LLM INFERENCE (Attention Kernel Throughput):")
    print(f"   FCSP: {fcsp_llm1:.2f} TFLOPS")
    print(f"   HAMi: {hami_llm1:.2f} TFLOPS")

    print("\n" + "="*100)

def main():
    benchmark_dir = os.path.dirname(os.path.abspath(__file__))

    # Load benchmark results
    files = {
        'native': 'native_results.json',
        'fcsp_adaptive': 'fcsp_adaptive_isolation_results.json',
        'hami_50mem_75sm': 'hami_50mem_75sm_results.json',
    }

    results_list = []
    names = []

    for name, filename in files.items():
        filepath = os.path.join(benchmark_dir, filename)
        if os.path.exists(filepath):
            results_list.append(load_results(filepath))
            names.append(name)
            print(f"Loaded: {filename}")
        else:
            print(f"Warning: {filename} not found")

    if len(results_list) >= 3:
        compare_metrics(results_list, names)
    else:
        print("Need at least 3 result files for comparison")

if __name__ == '__main__':
    main()
