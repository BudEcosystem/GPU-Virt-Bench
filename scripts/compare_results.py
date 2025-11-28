#!/usr/bin/env python3
"""
GPU Virtualization Benchmark Comparison Script

Compares benchmark results between different GPU virtualization systems
(e.g., FCSP vs HAMI vs Native).

Usage:
    python3 compare_results.py <result1.json> <result2.json> [result3.json ...]
    python3 compare_results.py --run-all  # Run benchmarks and compare
"""

import json
import sys
import os
import subprocess
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

@dataclass
class MetricResult:
    metric_id: str
    name: str
    value: float
    unit: str
    passed: bool
    category: str

@dataclass
class BenchmarkResult:
    system: str
    timestamp: str
    metrics: Dict[str, MetricResult]
    total_passed: int
    total_failed: int
    pass_rate: float

def load_results(filepath: str) -> Optional[BenchmarkResult]:
    """Load benchmark results from JSON file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        metrics = {}
        passed_count = 0
        failed_count = 0

        for m in data.get('metrics', []):
            # Handle both 'id' and 'metric_id' keys
            metric_id = m.get('id', m.get('metric_id', ''))
            if not metric_id:
                continue

            is_passed = m.get('success', m.get('passed', False))
            if is_passed:
                passed_count += 1
            else:
                failed_count += 1

            metrics[metric_id] = MetricResult(
                metric_id=metric_id,
                name=m.get('name', ''),
                value=m.get('value', 0.0),
                unit=m.get('unit', ''),
                passed=is_passed,
                category=metric_id.split('-')[0] if '-' in metric_id else 'UNKNOWN'
            )

        # Handle both formats for summary
        summary = data.get('summary', {})
        total_passed = summary.get('passed', passed_count)
        total_failed = summary.get('failed', failed_count)
        total = total_passed + total_failed
        pass_rate = (total_passed / total * 100) if total > 0 else 0.0

        # Handle both 'system' and 'system_name' keys
        system = data.get('system', data.get('system_name', 'unknown'))

        return BenchmarkResult(
            system=system,
            timestamp=data.get('timestamp', ''),
            metrics=metrics,
            total_passed=total_passed,
            total_failed=total_failed,
            pass_rate=pass_rate
        )
    except Exception as e:
        print(f"{Colors.RED}Error loading {filepath}: {e}{Colors.END}")
        import traceback
        traceback.print_exc()
        return None

def calculate_improvement(base_value: float, new_value: float, lower_is_better: bool = True) -> Tuple[float, str]:
    """Calculate percentage improvement between two values."""
    if base_value == 0:
        return 0.0, "N/A"

    if lower_is_better:
        # For latency, overhead - lower is better
        improvement = ((base_value - new_value) / base_value) * 100
    else:
        # For throughput, accuracy - higher is better
        improvement = ((new_value - base_value) / base_value) * 100

    if improvement > 0:
        return improvement, f"{Colors.GREEN}+{improvement:.1f}%{Colors.END}"
    elif improvement < 0:
        return improvement, f"{Colors.RED}{improvement:.1f}%{Colors.END}"
    else:
        return 0.0, f"{Colors.YELLOW}0.0%{Colors.END}"

def get_metric_direction(metric_id: str) -> bool:
    """Determine if lower values are better for this metric."""
    # Metrics where lower is better (latency, overhead, etc.)
    lower_is_better = [
        'OH-', 'LAT-', 'FRAG-', 'ERR-', 'SCHED-'
    ]
    # Metrics where higher is better (throughput, accuracy, etc.)
    higher_is_better = [
        'BW-', 'ISO-', 'LLM-', 'CACHE-', 'PCIE-', 'NCCL-', 'PAPER-'
    ]

    for prefix in lower_is_better:
        if metric_id.startswith(prefix):
            return True
    return False

def print_header(title: str):
    """Print a formatted header."""
    width = 100
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * width}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{title.center(width)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * width}{Colors.END}")
    print()

def print_section(title: str):
    """Print a section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{title}{Colors.END}")
    print(f"{Colors.BLUE}{'-' * len(title)}{Colors.END}")

def compare_results(results: List[BenchmarkResult], baseline_idx: int = 0):
    """Compare multiple benchmark results."""
    if len(results) < 2:
        print(f"{Colors.RED}Need at least 2 results to compare{Colors.END}")
        return

    baseline = results[baseline_idx]

    print_header("GPU Virtualization Benchmark Comparison")

    # Summary comparison
    print_section("Overall Summary")
    print(f"{'System':<15} {'Pass Rate':<12} {'Passed':<10} {'Failed':<10} {'Timestamp':<25}")
    print("-" * 72)
    for r in results:
        color = Colors.GREEN if r.pass_rate >= 80 else Colors.YELLOW if r.pass_rate >= 60 else Colors.RED
        print(f"{r.system:<15} {color}{r.pass_rate:>6.1f}%{Colors.END}     {r.total_passed:<10} {r.total_failed:<10} {r.timestamp:<25}")

    # Detailed metric comparison
    print_section("Metric-by-Metric Comparison")

    # Group metrics by category
    categories = {}
    all_metric_ids = set()
    for r in results:
        all_metric_ids.update(r.metrics.keys())

    for metric_id in sorted(all_metric_ids):
        category = metric_id.split('-')[0] if '-' in metric_id else 'OTHER'
        if category not in categories:
            categories[category] = []
        categories[category].append(metric_id)

    # Print each category
    for category in sorted(categories.keys()):
        print(f"\n{Colors.BOLD}{Colors.HEADER}Category: {category}{Colors.END}")

        # Header row
        header = f"{'Metric ID':<12} {'Name':<35}"
        for r in results:
            header += f" {r.system:<12}"
        if len(results) > 1:
            header += f" {'Improvement':<15}"
        print(header)
        print("-" * (60 + 13 * len(results)))

        for metric_id in sorted(categories[category]):
            # Get metric name from first result that has it
            name = ""
            for r in results:
                if metric_id in r.metrics:
                    name = r.metrics[metric_id].name[:33]
                    break

            row = f"{metric_id:<12} {name:<35}"

            values = []
            for r in results:
                if metric_id in r.metrics:
                    m = r.metrics[metric_id]
                    status = f"{Colors.GREEN}✓{Colors.END}" if m.passed else f"{Colors.RED}✗{Colors.END}"
                    row += f" {m.value:>8.2f}{status}   "
                    values.append(m.value)
                else:
                    row += f" {'N/A':>10}   "
                    values.append(None)

            # Calculate improvement vs baseline
            if len(values) >= 2 and values[0] is not None and values[1] is not None:
                lower_better = get_metric_direction(metric_id)
                _, improvement_str = calculate_improvement(values[0], values[1], lower_better)
                row += f" {improvement_str}"

            print(row)

    # Summary statistics
    print_section("Improvement Summary")

    improvements = {'better': 0, 'worse': 0, 'same': 0, 'na': 0}
    category_improvements = {}

    for metric_id in all_metric_ids:
        if metric_id not in baseline.metrics:
            improvements['na'] += 1
            continue

        other_values = []
        for i, r in enumerate(results):
            if i != baseline_idx and metric_id in r.metrics:
                other_values.append(r.metrics[metric_id].value)

        if not other_values:
            improvements['na'] += 1
            continue

        base_val = baseline.metrics[metric_id].value
        other_val = other_values[0]  # Compare with first non-baseline
        lower_better = get_metric_direction(metric_id)

        imp, _ = calculate_improvement(base_val, other_val, lower_better)

        category = metric_id.split('-')[0]
        if category not in category_improvements:
            category_improvements[category] = []
        category_improvements[category].append(imp)

        if abs(imp) < 1.0:
            improvements['same'] += 1
        elif imp > 0:
            improvements['better'] += 1
        else:
            improvements['worse'] += 1

    print(f"\nComparing {results[1].system} vs {baseline.system} (baseline):")
    print(f"  {Colors.GREEN}Better:{Colors.END}  {improvements['better']} metrics")
    print(f"  {Colors.RED}Worse:{Colors.END}   {improvements['worse']} metrics")
    print(f"  {Colors.YELLOW}Same:{Colors.END}    {improvements['same']} metrics")
    print(f"  N/A:     {improvements['na']} metrics")

    # Per-category average improvement
    print(f"\n{Colors.BOLD}Average Improvement by Category:{Colors.END}")
    for cat in sorted(category_improvements.keys()):
        imps = category_improvements[cat]
        avg_imp = sum(imps) / len(imps) if imps else 0
        color = Colors.GREEN if avg_imp > 0 else Colors.RED if avg_imp < 0 else Colors.YELLOW
        print(f"  {cat:<10}: {color}{avg_imp:+.1f}%{Colors.END} (across {len(imps)} metrics)")

def run_benchmark(system: str, output_dir: str, bench_binary: str) -> Optional[str]:
    """Run benchmark for a specific system."""
    output_file = os.path.join(output_dir, f"{system}_results.json")

    print(f"\n{Colors.BOLD}Running benchmark for {system}...{Colors.END}")

    cmd = [bench_binary, "--system", system, "--output", output_dir, "--json"]

    # For FCSP, we need to preload the library
    env = os.environ.copy()
    if system == "fcsp":
        fcsp_lib = "/home/bud/Desktop/hami/bud_fcsp/build/libvgpu.so"
        if os.path.exists(fcsp_lib):
            env["LD_PRELOAD"] = fcsp_lib
            env["BUD_MEMORY_LIMIT"] = "8589934592"  # 8GB
            env["BUD_COMPUTE_LIMIT"] = "100"
            print(f"  Using LD_PRELOAD={fcsp_lib}")
    elif system == "hami":
        hami_lib = "/home/bud/Desktop/hami/HAMi-core/build/libvgpu.so"
        if os.path.exists(hami_lib):
            env["LD_PRELOAD"] = hami_lib
            env["CUDA_DEVICE_MEMORY_LIMIT"] = "8589934592"
            env["CUDA_DEVICE_SM_LIMIT"] = "100"
            print(f"  Using LD_PRELOAD={hami_lib}")

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        if result.returncode != 0:
            print(f"{Colors.RED}Benchmark failed for {system}{Colors.END}")
            print(f"stdout: {result.stdout[:500]}")
            print(f"stderr: {result.stderr[:500]}")
            return None

        # Find the output JSON file
        json_files = list(Path(output_dir).glob(f"*{system}*.json"))
        if json_files:
            return str(max(json_files, key=os.path.getctime))

        return output_file if os.path.exists(output_file) else None

    except subprocess.TimeoutExpired:
        print(f"{Colors.RED}Benchmark timed out for {system}{Colors.END}")
        return None
    except Exception as e:
        print(f"{Colors.RED}Error running benchmark for {system}: {e}{Colors.END}")
        return None

def generate_report(results: List[BenchmarkResult], output_file: str):
    """Generate a markdown comparison report."""
    with open(output_file, 'w') as f:
        f.write("# GPU Virtualization Benchmark Comparison Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Systems compared
        f.write("## Systems Compared\n\n")
        for r in results:
            f.write(f"- **{r.system}**: {r.total_passed} passed, {r.total_failed} failed ({r.pass_rate:.1f}%)\n")
        f.write("\n")

        # Summary table
        f.write("## Summary\n\n")
        f.write("| Metric | " + " | ".join(r.system for r in results) + " |\n")
        f.write("|--------|" + "|".join(["--------"] * len(results)) + "|\n")
        f.write("| Pass Rate | " + " | ".join(f"{r.pass_rate:.1f}%" for r in results) + " |\n")
        f.write("| Passed | " + " | ".join(str(r.total_passed) for r in results) + " |\n")
        f.write("| Failed | " + " | ".join(str(r.total_failed) for r in results) + " |\n")
        f.write("\n")

        # Detailed metrics by category
        f.write("## Detailed Metrics\n\n")

        all_metric_ids = set()
        for r in results:
            all_metric_ids.update(r.metrics.keys())

        categories = {}
        for metric_id in sorted(all_metric_ids):
            category = metric_id.split('-')[0]
            if category not in categories:
                categories[category] = []
            categories[category].append(metric_id)

        for category in sorted(categories.keys()):
            f.write(f"### {category}\n\n")
            f.write("| Metric | Name | " + " | ".join(r.system for r in results) + " |\n")
            f.write("|--------|------|" + "|".join(["------"] * len(results)) + "|\n")

            for metric_id in sorted(categories[category]):
                name = ""
                for r in results:
                    if metric_id in r.metrics:
                        name = r.metrics[metric_id].name
                        break

                values = []
                for r in results:
                    if metric_id in r.metrics:
                        m = r.metrics[metric_id]
                        status = "✓" if m.passed else "✗"
                        values.append(f"{m.value:.2f} {status}")
                    else:
                        values.append("N/A")

                f.write(f"| {metric_id} | {name} | " + " | ".join(values) + " |\n")
            f.write("\n")

    print(f"\n{Colors.GREEN}Report saved to: {output_file}{Colors.END}")

def main():
    parser = argparse.ArgumentParser(description="Compare GPU virtualization benchmark results")
    parser.add_argument('files', nargs='*', help='JSON result files to compare')
    parser.add_argument('--run-all', action='store_true', help='Run benchmarks for native, hami, and fcsp')
    parser.add_argument('--run', nargs='+', choices=['native', 'hami', 'fcsp'], help='Run specific benchmarks')
    parser.add_argument('--output-dir', default='/home/bud/Desktop/hami/gpu-virt-bench/benchmarks', help='Output directory for results')
    parser.add_argument('--bench-binary', default='/home/bud/Desktop/hami/gpu-virt-bench/build/gpu-virt-bench', help='Path to benchmark binary')
    parser.add_argument('--report', help='Generate markdown report to specified file')

    args = parser.parse_args()

    result_files = []

    if args.run_all:
        args.run = ['native', 'hami', 'fcsp']

    if args.run:
        os.makedirs(args.output_dir, exist_ok=True)

        for system in args.run:
            result_file = run_benchmark(system, args.output_dir, args.bench_binary)
            if result_file:
                result_files.append(result_file)

    if args.files:
        result_files.extend(args.files)

    if not result_files:
        print(f"{Colors.RED}No result files to compare. Use --run or provide JSON files.{Colors.END}")
        parser.print_help()
        return 1

    # Load results
    results = []
    for f in result_files:
        r = load_results(f)
        if r:
            results.append(r)
            print(f"{Colors.GREEN}Loaded: {f} ({r.system}){Colors.END}")

    if len(results) < 1:
        print(f"{Colors.RED}No valid results loaded{Colors.END}")
        return 1

    if len(results) >= 2:
        compare_results(results)
    else:
        print(f"\n{Colors.YELLOW}Only one result loaded. Need at least 2 for comparison.{Colors.END}")
        print(f"System: {results[0].system}")
        print(f"Pass Rate: {results[0].pass_rate:.1f}%")
        print(f"Passed: {results[0].total_passed}, Failed: {results[0].total_failed}")

    if args.report and len(results) >= 2:
        generate_report(results, args.report)

    return 0

if __name__ == "__main__":
    sys.exit(main())
