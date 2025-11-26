# GPU Virtualization Performance Evaluation Tool (GPU-Virt-Bench)

A comprehensive benchmarking framework for evaluating software-based GPU virtualization systems like HAMi-core, BUD-FCSP, and comparing against ideal MIG behavior.

## Overview

This tool measures **56 performance metrics** across **10 categories** to provide comprehensive evaluation of GPU virtualization systems:

- **Overhead Metrics (10)**: Quantify the performance cost of virtualization
- **Isolation Metrics (10)**: Measure resource isolation quality between tenants
- **LLM Metrics (10)**: Evaluate performance for LLM inference workloads
- **Memory Bandwidth Metrics (4)**: Test bandwidth isolation and fairness
- **Cache Isolation Metrics (4)**: Measure L2 cache sharing impacts
- **PCIe Bandwidth Metrics (4)**: Evaluate host-device transfer performance
- **NCCL/P2P Communication Metrics (4)**: Test multi-GPU communication
- **Scheduling Metrics (4)**: Measure context switching and concurrency
- **Memory Fragmentation Metrics (3)**: Track memory fragmentation impacts
- **Error Recovery Metrics (3)**: Test error handling and fault isolation

## Supported Virtualization Systems

| System | Description | Config Key |
|--------|-------------|------------|
| `native` | No virtualization (baseline) | `native` |
| `hami-core` | HAMi-core original implementation | `hami` |
| `fcsp` | BUD-FCSP improved implementation | `fcsp` |
| `mig-ideal` | Simulated ideal MIG behavior | `mig` |

## Metrics Reference

### Category 1: Overhead Metrics (OH-001 to OH-010)

These metrics quantify the performance overhead introduced by virtualization layer.

| Metric ID | Name | Description | Unit | Lower is Better |
|-----------|------|-------------|------|-----------------|
| `OH-001` | Kernel Launch Latency | Time from cuLaunchKernel call to kernel execution start | μs | ✓ |
| `OH-002` | Memory Allocation Latency | Time for cuMemAlloc to complete | μs | ✓ |
| `OH-003` | Memory Free Latency | Time for cuMemFree to complete | μs | ✓ |
| `OH-004` | Context Creation Overhead | Additional time for context creation vs native | μs | ✓ |
| `OH-005` | API Interception Overhead | dlsym hook overhead per CUDA call | ns | ✓ |
| `OH-006` | Shared Region Lock Contention | Time waiting for shared region semaphore | μs | ✓ |
| `OH-007` | Memory Tracking Overhead | Per-allocation memory accounting cost | ns | ✓ |
| `OH-008` | Rate Limiter Overhead | Token bucket check latency | ns | ✓ |
| `OH-009` | NVML Polling Overhead | CPU cycles spent in utilization monitoring | % | ✓ |
| `OH-010` | Total Throughput Degradation | End-to-end performance vs native | % | ✓ |

### Category 2: Isolation Metrics (IS-001 to IS-010)

These metrics evaluate how well the system isolates resources between tenants.

| Metric ID | Name | Description | Unit | Higher is Better |
|-----------|------|-------------|------|------------------|
| `IS-001` | Memory Limit Accuracy | Actual vs configured memory limit | % | ✓ |
| `IS-002` | Memory Limit Enforcement | Time to detect/block over-allocation | μs | ✗ |
| `IS-003` | SM Utilization Accuracy | Actual vs configured SM limit | % | ✓ |
| `IS-004` | SM Limit Response Time | Time to adjust utilization after limit change | ms | ✗ |
| `IS-005` | Cross-Tenant Memory Isolation | Memory leak detection between containers | boolean | ✓ |
| `IS-006` | Cross-Tenant Compute Isolation | Compute interference measurement | % | ✓ |
| `IS-007` | QoS Consistency | Variance in performance under contention | CV | ✗ |
| `IS-008` | Fairness Index | Jain's fairness index across tenants | 0-1 | ✓ |
| `IS-009` | Noisy Neighbor Impact | Performance degradation from aggressive neighbor | % | ✗ |
| `IS-010` | Fault Isolation | Error propagation between containers | boolean | ✓ |

### Category 3: LLM-Friendly Metrics (LLM-001 to LLM-010)

These metrics target LLM inference and training workloads.

| Metric ID | Name | Description | Unit | Higher is Better |
|-----------|------|-------------|------|------------------|
| `LLM-001` | Attention Kernel Throughput | Transformer attention performance | TFLOPS | ✓ |
| `LLM-002` | KV Cache Allocation Speed | Dynamic KV cache growth handling | allocs/sec | ✓ |
| `LLM-003` | Batch Size Scaling | Throughput vs batch size curve | ratio | ✓ |
| `LLM-004` | Token Generation Latency | Time-to-first-token and inter-token latency | ms | ✗ |
| `LLM-005` | Memory Pool Efficiency | Pool-based allocation overhead | % | ✗ |
| `LLM-006` | Multi-Stream Performance | Pipeline parallel efficiency | % | ✓ |
| `LLM-007` | Large Tensor Allocation | Large contiguous allocation handling | ms | ✗ |
| `LLM-008` | Mixed Precision Support | FP16/BF16 kernel handling | ratio | ✓ |
| `LLM-009` | Dynamic Batching Impact | Variable batch handling | latency variance | ✗ |
| `LLM-010` | Multi-GPU Scaling | Tensor parallel efficiency | scaling factor | ✓ |

### Category 4: Memory Bandwidth Metrics (BW-001 to BW-004)

These metrics measure memory bandwidth isolation and fairness.

| Metric ID | Name | Description | Unit | Higher is Better |
|-----------|------|-------------|------|------------------|
| `BW-001` | Memory Bandwidth Isolation | Bandwidth achieved under contention | % | ✓ |
| `BW-002` | Bandwidth Fairness Index | Jain's fairness for bandwidth distribution | ratio | ✓ |
| `BW-003` | Memory Bus Saturation Point | Concurrent streams before saturation | ratio | ✓ |
| `BW-004` | Bandwidth Interference Impact | Bandwidth drop from competing workloads | % | ✗ |

### Category 5: Cache Isolation Metrics (CACHE-001 to CACHE-004)

These metrics evaluate L2 cache sharing and isolation.

| Metric ID | Name | Description | Unit | Higher is Better |
|-----------|------|-------------|------|------------------|
| `CACHE-001` | L2 Cache Hit Rate | Cache hit rate under multi-tenant workloads | % | ✓ |
| `CACHE-002` | Cache Eviction Rate | Evictions caused by other tenants | % | ✗ |
| `CACHE-003` | Working Set Collision Impact | Performance drop from cache overlap | % | ✗ |
| `CACHE-004` | Cache Contention Overhead | Additional latency from L2 contention | % | ✗ |

### Category 6: PCIe Bandwidth Metrics (PCIE-001 to PCIE-004)

These metrics test host-device data transfer performance.

| Metric ID | Name | Description | Unit | Higher is Better |
|-----------|------|-------------|------|------------------|
| `PCIE-001` | Host-to-Device Bandwidth | H2D bandwidth as % of theoretical max | GB/s | ✓ |
| `PCIE-002` | Device-to-Host Bandwidth | D2H bandwidth as % of theoretical max | GB/s | ✓ |
| `PCIE-003` | PCIe Contention Impact | Bandwidth drop under multi-tenant traffic | % | ✗ |
| `PCIE-004` | Pinned Memory Performance | Pinned vs pageable memory transfer ratio | ratio | ✓ |

### Category 7: NCCL/P2P Communication Metrics (NCCL-001 to NCCL-004)

These metrics evaluate multi-GPU communication for distributed training.

| Metric ID | Name | Description | Unit | Higher is Better |
|-----------|------|-------------|------|------------------|
| `NCCL-001` | AllReduce Latency | Time for allreduce collective operation | μs | ✗ |
| `NCCL-002` | AllGather Bandwidth | Achieved bandwidth for allgather | GB/s | ✓ |
| `NCCL-003` | P2P GPU Bandwidth | Direct GPU-to-GPU bandwidth (NVLink/PCIe) | GB/s | ✓ |
| `NCCL-004` | Broadcast Bandwidth | Achieved bandwidth for broadcast | GB/s | ✓ |

### Category 8: Scheduling Metrics (SCHED-001 to SCHED-004)

These metrics measure kernel scheduling and concurrency.

| Metric ID | Name | Description | Unit | Higher is Better |
|-----------|------|-------------|------|------------------|
| `SCHED-001` | Context Switch Latency | Time to switch between CUDA contexts | μs | ✗ |
| `SCHED-002` | Kernel Launch Overhead | Overhead of launching minimal kernels | μs | ✗ |
| `SCHED-003` | Stream Concurrency Efficiency | Concurrent stream execution efficiency | % | ✓ |
| `SCHED-004` | Preemption Latency | Latency when higher priority work preempts | ms | ✗ |

### Category 9: Memory Fragmentation Metrics (FRAG-001 to FRAG-003)

These metrics track memory fragmentation and its impacts.

| Metric ID | Name | Description | Unit | Higher is Better |
|-----------|------|-------------|------|------------------|
| `FRAG-001` | Fragmentation Index | Memory fragmentation after alloc/free cycles | % | ✗ |
| `FRAG-002` | Allocation Latency Degradation | Allocation latency increase with fragmentation | ratio | ✗ |
| `FRAG-003` | Memory Compaction Efficiency | Memory reclaimed after defragmentation | % | ✓ |

### Category 10: Error Recovery Metrics (ERR-001 to ERR-003)

These metrics test error handling and fault isolation.

| Metric ID | Name | Description | Unit | Higher is Better |
|-----------|------|-------------|------|------------------|
| `ERR-001` | Error Detection Latency | Time to detect and report CUDA errors | μs | ✗ |
| `ERR-002` | Error Recovery Time | Time to recover GPU to usable state | μs | ✗ |
| `ERR-003` | Graceful Degradation Score | Handling resource exhaustion without crashing | % | ✓ |

## Installation

### Prerequisites

- **CUDA Toolkit** 11.0+ (with nvcc compiler)
  - Check: `nvcc --version`
- **NVIDIA GPU** with compute capability 7.0+ (V100, T4, A100, RTX 30xx/40xx series, etc.)
  - Check: `nvidia-smi`
- **Linux** (tested on Ubuntu 20.04/22.04)
- **GCC** 9+ or compatible C compiler
  - Check: `gcc --version`
- **CMake** 3.18+ (recommended) or Make
  - Check: `cmake --version`

### Build with CMake (Recommended)

```bash
cd gpu-virt-bench
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

The binary will be created at `build/gpu-virt-bench`.

### Build with Make

```bash
cd gpu-virt-bench
make
```

### Troubleshooting Build Issues

**Issue**: `nvcc: command not found`
- **Solution**: Add CUDA to PATH: `export PATH=/usr/local/cuda/bin:$PATH`

**Issue**: `Cannot find -lcuda` or `-lcudart`
- **Solution**: Add CUDA libs: `export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH`

**Issue**: CMake cannot find CUDA
- **Solution**: Set CUDA path explicitly: `cmake -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda ..`

## Usage

### Command Line Options

```
./gpu-virt-bench --system <name> [options]

Required:
  --system <name>         System to benchmark:
                            native - Bare metal (no virtualization)
                            hami   - HAMi-core virtualization
                            fcsp   - BUD-FCSP improved virtualization

Options:
  --output <dir>          Output directory (default: ./benchmarks)
  --iterations <n>        Benchmark iterations (default: 100)
  --warmup <n>            Warmup iterations (default: 10)
  --metrics <list>        Comma-separated metric IDs to run
                          (default: all, e.g., OH-001,IS-005,LLM-003)
  --processes <n>         Concurrent processes for isolation tests (default: 4)
  --memory-limit <MB>     Memory limit to test (default: 4096)
  --compute-limit <%>     Compute limit percentage (default: 50)
  --compare <file>        JSON file to compare results against
  --verbose               Verbose output
  --json-only             Only output JSON (skip text reports)
  --help                  Show help message
```

### Quick Start Examples

#### Basic Benchmarks

```bash
# Run all benchmarks on native (baseline)
./build/gpu-virt-bench --system native

# Run HAMi-core benchmarks with custom limits
./build/gpu-virt-bench --system hami --memory-limit 2048 --compute-limit 30

# Run BUD-FCSP benchmarks
./build/gpu-virt-bench --system fcsp
```

#### Category-Specific Benchmarks

```bash
# Run only overhead metrics
./build/gpu-virt-bench --system hami --metrics OH-001,OH-002,OH-003,OH-004,OH-005,OH-006,OH-007,OH-008,OH-009,OH-010

# Run only LLM metrics
./build/gpu-virt-bench --system fcsp --metrics LLM-001,LLM-002,LLM-003,LLM-004,LLM-005,LLM-006,LLM-007,LLM-008,LLM-009,LLM-010

# Run memory bandwidth metrics
./build/gpu-virt-bench --system hami --metrics BW-001,BW-002,BW-003,BW-004

# Run cache isolation metrics
./build/gpu-virt-bench --system hami --metrics CACHE-001,CACHE-002,CACHE-003,CACHE-004

# Run PCIe bandwidth metrics
./build/gpu-virt-bench --system native --metrics PCIE-001,PCIE-002,PCIE-003,PCIE-004

# Run NCCL/P2P communication metrics
./build/gpu-virt-bench --system hami --metrics NCCL-001,NCCL-002,NCCL-003,NCCL-004

# Run scheduling metrics
./build/gpu-virt-bench --system hami --metrics SCHED-001,SCHED-002,SCHED-003,SCHED-004

# Run fragmentation metrics
./build/gpu-virt-bench --system hami --metrics FRAG-001,FRAG-002,FRAG-003

# Run error recovery metrics
./build/gpu-virt-bench --system hami --metrics ERR-001,ERR-002,ERR-003
```

#### Comparison and Analysis

```bash
# Compare FCSP against HAMi-core baseline
./build/gpu-virt-bench --system fcsp --compare benchmarks/hami_results.json

# Run specific metrics only with verbose output
./build/gpu-virt-bench --system native --metrics OH-001,BW-001,LLM-001 --verbose

# Run complete performance profile
./build/gpu-virt-bench --system hami --iterations 200 --warmup 20
```

### Running Benchmarks by Category

#### Using Make Targets

```bash
make benchmark-native   # Run native benchmark
make benchmark-hami     # Run HAMi-core benchmark
make benchmark-fcsp     # Run BUD-FCSP benchmark
make benchmark-all      # Run all benchmarks
make test               # Run quick validation test
make clean              # Clean build artifacts
```

#### Using Test Scripts

```bash
# Run quick validation tests
./scripts/run_tests.sh

# Run specific test categories
./scripts/run_tests.sh basic overhead llm

# Run full benchmark suite (all metrics, takes longer)
./scripts/run_tests.sh full

# Available test categories:
# - help      : Test help message
# - basic     : Basic execution test
# - multiple  : Multiple metrics test
# - overhead  : All overhead metrics (OH-001 to OH-010)
# - llm       : LLM metrics (LLM-001 to LLM-010)
# - json      : JSON output only test
# - verbose   : Verbose output test
# - limits    : Custom limits test
# - reports   : Report generation test
# - full      : Full benchmark (all 56 metrics)
```

### Multi-Tenant Testing

Test isolation with multiple concurrent tenants:

```bash
# Run multi-tenant isolation test (spawns multiple processes)
NUM_TENANTS=4 SYSTEM=hami ./scripts/multi_tenant_test.sh

# With custom configuration
NUM_TENANTS=8 MEMORY_PER_TENANT=1024 COMPUTE_PER_TENANT=12 ./scripts/multi_tenant_test.sh
```

## Output and Results

### Output Directory Structure

Results are saved to `benchmarks/` directory organized by system:

```
benchmarks/
├── native/
│   ├── native_20241126_130000.json   # JSON report (machine-readable)
│   ├── native_20241126_130000.csv    # CSV data (for analysis)
│   └── native_20241126_130000.txt    # Human-readable report
├── hami/
│   ├── hami_20241126_130500.json
│   ├── hami_20241126_130500.csv
│   └── hami_20241126_130500.txt
├── fcsp/
│   └── ...
└── comparison_hami_vs_fcsp_20241126_131000.txt  # Comparison report
```

### Report Formats

**JSON Report** - Machine-readable format with full statistical data:
```json
{
  "benchmark_version": "1.0.0",
  "timestamp": "2024-11-26 13:00:00",
  "system": {"name": "hami", "version": "1.0.0"},
  "metrics": [
    {
      "id": "OH-001",
      "name": "Kernel Launch Latency",
      "statistics": {"mean": 15.3, "p99": 45.2, "stddev": 8.2},
      "mig_comparison": {"mig_expected": 5.0, "mig_gap_percent": 206.0}
    }
  ],
  "summary": {"overall_score": 0.72, "mig_parity_percent": 72.0}
}
```

**Text Report** - Human-readable summary with grades:
```
================================================================================
  BENCHMARK SUMMARY: hami v1.0.0
================================================================================

  Overall Score:    72.0% - C  (Fair)
  MIG Parity:       72.0%

  Category Scores:
    Overhead:       65.0%
    Isolation:      78.0%
    LLM:            73.0%
    Bandwidth:      70.0%
    Cache:          68.0%
    PCIe:           85.0%
    NCCL:           75.0%
    Scheduling:     66.0%
    Fragmentation:  62.0%
    Error Recovery: 80.0%
```

### Scoring System

- **A+ (>= 95%)**: Excellent - Near MIG performance
- **A  (>= 90%)**: Very Good
- **B+ (>= 85%)**: Good
- **B  (>= 80%)**: Acceptable
- **C  (>= 70%)**: Fair
- **D  (>= 60%)**: Poor
- **F  (< 60%)**: Failing - Significant improvement needed

### Interpreting Results

**For Overhead Metrics**: Lower values indicate better performance. Compare against native baseline to quantify virtualization overhead.

**For Isolation Metrics**: Higher accuracy and lower latency values indicate better isolation. Boolean metrics (IS-005, IS-010) should be 1.0 (true) for proper isolation.

**For LLM Metrics**: Higher throughput and lower latency values indicate better LLM performance. Scaling ratios should approach 1.0 for linear scaling.

**For Extended Metrics**: Each category has specific benchmarks for bandwidth, cache, PCIe, NCCL, scheduling, fragmentation, and error recovery. Refer to metric descriptions for interpretation guidance.

## Architecture

### Directory Structure

```
gpu-virt-bench/
├── src/
│   ├── main.c                 # Entry point and CLI parsing
│   ├── metrics/
│   │   ├── overhead.cu        # Overhead benchmarks (OH-001 to OH-010)
│   │   ├── isolation.cu       # Isolation benchmarks (IS-001 to IS-010)
│   │   ├── llm.cu             # LLM benchmarks (LLM-001 to LLM-010)
│   │   ├── bandwidth.cu       # Memory bandwidth (BW-001 to BW-004)
│   │   ├── cache.cu           # Cache isolation (CACHE-001 to CACHE-004)
│   │   ├── pcie.cu            # PCIe bandwidth (PCIE-001 to PCIE-004)
│   │   ├── nccl.cu            # NCCL/P2P comm (NCCL-001 to NCCL-004)
│   │   ├── scheduling.cu      # Scheduling (SCHED-001 to SCHED-004)
│   │   ├── fragmentation.cu   # Fragmentation (FRAG-001 to FRAG-003)
│   │   └── error.cu           # Error recovery (ERR-001 to ERR-003)
│   ├── utils/
│   │   ├── timing.c           # High-precision timing utilities
│   │   ├── statistics.c       # Statistical analysis (mean, p99, CV, etc.)
│   │   ├── report.c           # Result reporting and formatting
│   │   └── process.c          # Multi-process test orchestration
│   └── include/
│       ├── benchmark.h        # Core benchmark types and definitions
│       └── metrics.h          # Metric registry and definitions
├── configs/
│   └── default.conf           # Default configuration
├── scripts/
│   ├── run_tests.sh           # Test suite runner
│   └── multi_tenant_test.sh   # Multi-tenant testing
├── benchmarks/                # Benchmark results (JSON, CSV, TXT)
├── results/                   # Additional result files
├── build/                     # Build output directory
├── CMakeLists.txt             # CMake build configuration
├── Makefile                   # Make build configuration
└── README.md                  # This file
```

### Key Components

**Metric Implementation**: Each `.cu` file in `src/metrics/` implements a category of benchmarks using CUDA kernels and host code.

**Multi-Process Testing**: The `process.c` utility enables spawning multiple concurrent processes to test isolation under contention.

**Statistical Analysis**: The `statistics.c` module computes mean, median, percentiles (p50, p95, p99), standard deviation, and coefficient of variation.

**MIG Comparison**: Each metric has an expected MIG baseline value for comparison. The tool calculates a "MIG gap" percentage showing how close the virtualization system comes to hardware MIG performance.

## Performance Tuning Tips

1. **Increase iterations** for more stable results: `--iterations 500`
2. **Use warmup runs** to avoid cold-start effects: `--warmup 20`
3. **Run overnight** for comprehensive benchmarks covering all 56 metrics
4. **Isolate the test machine** from other workloads for consistent results
5. **Use performance mode**: `sudo nvidia-smi -pm 1` and disable GPU boost variations
6. **Monitor with nvidia-smi** during tests: `watch -n 1 nvidia-smi`

## Contributing

Contributions are welcome! To add new metrics:

1. Add metric definition to `src/include/metrics.h`
2. Implement benchmark in appropriate `src/metrics/*.cu` file
3. Add MIG expected value if applicable
4. Update this README with metric description
5. Add test case to `scripts/run_tests.sh`

## License

Apache 2.0

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{gpu_virt_bench,
  title = {GPU-Virt-Bench: Comprehensive GPU Virtualization Benchmarking Tool},
  author = {BUD Ecosystem Team},
  year = {2024},
  url = {https://github.com/BudEcosystem/GPU-Virt-Bench}
}
```

## Related Projects

- [HAMi](https://github.com/Project-HAMi/HAMi) - Heterogeneous AI Computing Virtualization Middleware
- [BUD-FCSP](https://github.com/BudEcosystem) - Fine-grained Container-level SM Partitioning

## Support

For issues, questions, or feature requests, please open an issue on [GitHub](https://github.com/BudEcosystem/GPU-Virt-Bench/issues).
