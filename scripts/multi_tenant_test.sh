#!/bin/bash
#
# Multi-Tenant Isolation Test
# Spawns multiple processes to test isolation and fairness
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
BENCHMARK_DIR="$PROJECT_DIR/benchmarks/multi_tenant"
EXECUTABLE="$BUILD_DIR/gpu-virt-bench"

# Default configuration
NUM_TENANTS=${NUM_TENANTS:-4}
ITERATIONS=${ITERATIONS:-100}
MEMORY_PER_TENANT=${MEMORY_PER_TENANT:-2048}  # MB
COMPUTE_PER_TENANT=${COMPUTE_PER_TENANT:-25}   # %
SYSTEM=${SYSTEM:-native}

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo ""
    echo -e "${BLUE}================================================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}================================================================================${NC}"
    echo ""
}

# Check executable
if [ ! -f "$EXECUTABLE" ]; then
    echo "Building benchmark tool..."
    cd "$PROJECT_DIR"
    make
fi

mkdir -p "$BENCHMARK_DIR"

print_header "Multi-Tenant Isolation Test"

echo "Configuration:"
echo "  System:              $SYSTEM"
echo "  Number of tenants:   $NUM_TENANTS"
echo "  Iterations:          $ITERATIONS"
echo "  Memory per tenant:   $MEMORY_PER_TENANT MB"
echo "  Compute per tenant:  $COMPUTE_PER_TENANT%"
echo ""

# Array to hold PIDs
declare -a PIDS
declare -a OUTPUT_FILES

# Start benchmark processes
print_header "Starting $NUM_TENANTS Concurrent Benchmark Processes"

for i in $(seq 1 $NUM_TENANTS); do
    OUTPUT_FILE="$BENCHMARK_DIR/tenant_${i}_$(date +%Y%m%d_%H%M%S).log"
    OUTPUT_FILES+=("$OUTPUT_FILE")

    echo "Starting tenant $i..."

    "$EXECUTABLE" \
        --system "$SYSTEM" \
        --iterations "$ITERATIONS" \
        --warmup 5 \
        --metrics IS-001,IS-003,IS-008,LLM-001,LLM-009 \
        --memory-limit "$MEMORY_PER_TENANT" \
        --compute-limit "$COMPUTE_PER_TENANT" \
        --output "$BENCHMARK_DIR/tenant_$i" \
        > "$OUTPUT_FILE" 2>&1 &

    PIDS+=($!)
    echo "  PID: ${PIDS[-1]}"
done

echo ""
echo "All processes started. Waiting for completion..."
echo ""

# Monitor progress
RUNNING=${#PIDS[@]}
while [ $RUNNING -gt 0 ]; do
    RUNNING=0
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            ((RUNNING++))
        fi
    done

    if [ $RUNNING -gt 0 ]; then
        echo -ne "\r  Running: $RUNNING / ${#PIDS[@]} processes    "
        sleep 2
    fi
done

echo -e "\n"
print_header "Results Collection"

# Collect and analyze results
declare -a SCORES
declare -a THROUGHPUTS

for i in $(seq 0 $((NUM_TENANTS - 1))); do
    tenant_num=$((i + 1))
    log_file="${OUTPUT_FILES[$i]}"

    if [ -f "$log_file" ]; then
        # Extract overall score from log
        score=$(grep -oP "Overall Score:\s+\K[\d.]+" "$log_file" 2>/dev/null || echo "0")
        SCORES+=("$score")

        echo "Tenant $tenant_num:"
        echo "  Log: $log_file"
        echo "  Overall Score: $score%"

        # Show key metrics if available
        grep -E "(IS-00[138]|LLM-00[19])" "$log_file" 2>/dev/null | head -5 || true
        echo ""
    else
        echo -e "${RED}Tenant $tenant_num: Log file not found${NC}"
        SCORES+=("0")
    fi
done

# Calculate fairness
print_header "Fairness Analysis"

if [ ${#SCORES[@]} -gt 1 ]; then
    # Calculate mean
    sum=0
    for score in "${SCORES[@]}"; do
        sum=$(echo "$sum + $score" | bc -l)
    done
    mean=$(echo "scale=2; $sum / ${#SCORES[@]}" | bc -l)

    # Calculate variance
    variance_sum=0
    for score in "${SCORES[@]}"; do
        diff=$(echo "$score - $mean" | bc -l)
        sq=$(echo "$diff * $diff" | bc -l)
        variance_sum=$(echo "$variance_sum + $sq" | bc -l)
    done
    std_dev=$(echo "scale=2; sqrt($variance_sum / ${#SCORES[@]})" | bc -l)

    # Calculate Jain's Fairness Index
    sum_sq=0
    for score in "${SCORES[@]}"; do
        sq=$(echo "$score * $score" | bc -l)
        sum_sq=$(echo "$sum_sq + $sq" | bc -l)
    done
    jains=$(echo "scale=4; ($sum * $sum) / (${#SCORES[@]} * $sum_sq)" | bc -l)

    echo "Score Statistics:"
    echo "  Mean Score:           $mean%"
    echo "  Standard Deviation:   $std_dev"
    echo "  Jain's Fairness:      $jains (1.0 = perfectly fair)"
    echo ""

    # Determine fairness grade
    if (( $(echo "$jains >= 0.95" | bc -l) )); then
        echo -e "${GREEN}Fairness Grade: EXCELLENT (>= 0.95)${NC}"
    elif (( $(echo "$jains >= 0.90" | bc -l) )); then
        echo -e "${GREEN}Fairness Grade: GOOD (>= 0.90)${NC}"
    elif (( $(echo "$jains >= 0.80" | bc -l) )); then
        echo -e "${YELLOW}Fairness Grade: ACCEPTABLE (>= 0.80)${NC}"
    else
        echo -e "${RED}Fairness Grade: POOR (< 0.80)${NC}"
    fi
fi

print_header "Detailed Reports"
echo "Reports saved to: $BENCHMARK_DIR"
echo ""
ls -la "$BENCHMARK_DIR"/ 2>/dev/null || echo "No report files found"

echo ""
echo "Test complete."
