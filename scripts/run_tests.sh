#!/bin/bash
#
# GPU Virtualization Benchmark Test Suite
# Runs various test scenarios to validate the benchmark tool
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
BENCHMARK_DIR="$PROJECT_DIR/benchmarks"
EXECUTABLE="$BUILD_DIR/gpu-virt-bench"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo "================================================================================"
    echo "  $1"
    echo "================================================================================"
    echo ""
}

print_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

print_failure() {
    echo -e "${RED}[FAIL]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Check if executable exists
check_build() {
    if [ ! -f "$EXECUTABLE" ]; then
        echo "Executable not found at $EXECUTABLE"
        echo "Building..."
        cd "$PROJECT_DIR"
        if [ -f "CMakeLists.txt" ]; then
            mkdir -p build && cd build
            cmake .. && make -j$(nproc)
        else
            make
        fi
    fi
}

# Test 1: Help message
test_help() {
    print_header "Test 1: Help Message"

    if "$EXECUTABLE" --help > /dev/null 2>&1; then
        print_success "Help message displayed correctly"
        return 0
    else
        print_failure "Help message failed"
        return 1
    fi
}

# Test 2: Basic execution with minimal iterations
test_basic_execution() {
    print_header "Test 2: Basic Execution (Native, 5 iterations)"

    if "$EXECUTABLE" --system native --iterations 5 --warmup 1 \
                     --metrics OH-001 --output "$BENCHMARK_DIR/test" 2>&1; then
        print_success "Basic execution completed"
        return 0
    else
        print_failure "Basic execution failed"
        return 1
    fi
}

# Test 3: Multiple metrics
test_multiple_metrics() {
    print_header "Test 3: Multiple Metrics"

    if "$EXECUTABLE" --system native --iterations 5 --warmup 1 \
                     --metrics OH-001,OH-002,OH-003 \
                     --output "$BENCHMARK_DIR/test" 2>&1; then
        print_success "Multiple metrics test completed"
        return 0
    else
        print_failure "Multiple metrics test failed"
        return 1
    fi
}

# Test 4: All overhead metrics
test_overhead_metrics() {
    print_header "Test 4: All Overhead Metrics (OH-001 to OH-010)"

    if "$EXECUTABLE" --system native --iterations 10 --warmup 2 \
                     --metrics OH-001,OH-002,OH-003,OH-004,OH-005,OH-006,OH-007,OH-008,OH-009,OH-010 \
                     --output "$BENCHMARK_DIR/test" 2>&1; then
        print_success "Overhead metrics test completed"
        return 0
    else
        print_failure "Overhead metrics test failed"
        return 1
    fi
}

# Test 5: LLM metrics
test_llm_metrics() {
    print_header "Test 5: LLM Metrics (LLM-001 to LLM-005)"

    if "$EXECUTABLE" --system native --iterations 10 --warmup 2 \
                     --metrics LLM-001,LLM-002,LLM-003,LLM-004,LLM-005 \
                     --output "$BENCHMARK_DIR/test" 2>&1; then
        print_success "LLM metrics test completed"
        return 0
    else
        print_failure "LLM metrics test failed"
        return 1
    fi
}

# Test 6: JSON output only
test_json_output() {
    print_header "Test 6: JSON Output Only"

    if "$EXECUTABLE" --system native --iterations 5 --warmup 1 \
                     --metrics OH-001 --json-only \
                     --output "$BENCHMARK_DIR/test" 2>&1; then
        # Check if JSON file was created
        if ls "$BENCHMARK_DIR/test/native/"*.json 1>/dev/null 2>&1; then
            print_success "JSON output created successfully"
            return 0
        else
            print_failure "JSON file not found"
            return 1
        fi
    else
        print_failure "JSON output test failed"
        return 1
    fi
}

# Test 7: Verbose output
test_verbose() {
    print_header "Test 7: Verbose Output"

    if "$EXECUTABLE" --system native --iterations 5 --warmup 1 \
                     --metrics OH-001 --verbose \
                     --output "$BENCHMARK_DIR/test" 2>&1; then
        print_success "Verbose output test completed"
        return 0
    else
        print_failure "Verbose output test failed"
        return 1
    fi
}

# Test 8: Custom limits
test_custom_limits() {
    print_header "Test 8: Custom Memory and Compute Limits"

    if "$EXECUTABLE" --system native --iterations 5 --warmup 1 \
                     --metrics OH-001,IS-001 \
                     --memory-limit 2048 --compute-limit 30 \
                     --output "$BENCHMARK_DIR/test" 2>&1; then
        print_success "Custom limits test completed"
        return 0
    else
        print_failure "Custom limits test failed"
        return 1
    fi
}

# Test 9: Report generation
test_report_generation() {
    print_header "Test 9: Report Generation (JSON, CSV, Text)"

    # Clean test output
    rm -rf "$BENCHMARK_DIR/test/native" 2>/dev/null || true

    if "$EXECUTABLE" --system native --iterations 10 --warmup 2 \
                     --metrics OH-001,OH-002,LLM-001 \
                     --output "$BENCHMARK_DIR/test" 2>&1; then

        # Check for all report types
        local json_count=$(ls "$BENCHMARK_DIR/test/native/"*.json 2>/dev/null | wc -l)
        local csv_count=$(ls "$BENCHMARK_DIR/test/native/"*.csv 2>/dev/null | wc -l)
        local txt_count=$(ls "$BENCHMARK_DIR/test/native/"*.txt 2>/dev/null | wc -l)

        if [ "$json_count" -gt 0 ] && [ "$csv_count" -gt 0 ] && [ "$txt_count" -gt 0 ]; then
            print_success "All report formats generated"
            echo "  - JSON files: $json_count"
            echo "  - CSV files: $csv_count"
            echo "  - Text files: $txt_count"
            return 0
        else
            print_failure "Missing report files"
            return 1
        fi
    else
        print_failure "Report generation test failed"
        return 1
    fi
}

# Full benchmark test (takes longer)
test_full_benchmark() {
    print_header "Test 10: Full Native Benchmark (All Metrics)"

    echo "This test runs all 30 metrics and may take several minutes..."

    if "$EXECUTABLE" --system native --iterations 50 --warmup 10 \
                     --output "$BENCHMARK_DIR/full_test" 2>&1; then
        print_success "Full benchmark completed"
        return 0
    else
        print_failure "Full benchmark failed"
        return 1
    fi
}

# Main test runner
main() {
    print_header "GPU Virtualization Benchmark Test Suite"

    # Create test output directory
    mkdir -p "$BENCHMARK_DIR/test"

    # Check/build executable
    check_build

    local passed=0
    local failed=0
    local tests=()

    # Parse arguments
    if [ $# -eq 0 ]; then
        tests=("help" "basic" "multiple" "overhead" "llm" "json" "verbose" "limits" "reports")
    else
        tests=("$@")
    fi

    for test in "${tests[@]}"; do
        case "$test" in
            "help")
                test_help && ((passed++)) || ((failed++))
                ;;
            "basic")
                test_basic_execution && ((passed++)) || ((failed++))
                ;;
            "multiple")
                test_multiple_metrics && ((passed++)) || ((failed++))
                ;;
            "overhead")
                test_overhead_metrics && ((passed++)) || ((failed++))
                ;;
            "llm")
                test_llm_metrics && ((passed++)) || ((failed++))
                ;;
            "json")
                test_json_output && ((passed++)) || ((failed++))
                ;;
            "verbose")
                test_verbose && ((passed++)) || ((failed++))
                ;;
            "limits")
                test_custom_limits && ((passed++)) || ((failed++))
                ;;
            "reports")
                test_report_generation && ((passed++)) || ((failed++))
                ;;
            "full")
                test_full_benchmark && ((passed++)) || ((failed++))
                ;;
            "all")
                test_help && ((passed++)) || ((failed++))
                test_basic_execution && ((passed++)) || ((failed++))
                test_multiple_metrics && ((passed++)) || ((failed++))
                test_overhead_metrics && ((passed++)) || ((failed++))
                test_llm_metrics && ((passed++)) || ((failed++))
                test_json_output && ((passed++)) || ((failed++))
                test_verbose && ((passed++)) || ((failed++))
                test_custom_limits && ((passed++)) || ((failed++))
                test_report_generation && ((passed++)) || ((failed++))
                test_full_benchmark && ((passed++)) || ((failed++))
                ;;
            *)
                print_warning "Unknown test: $test"
                ;;
        esac
    done

    # Summary
    print_header "Test Summary"
    echo "  Passed: $passed"
    echo "  Failed: $failed"
    echo ""

    if [ $failed -eq 0 ]; then
        echo -e "${GREEN}All tests passed!${NC}"
        return 0
    else
        echo -e "${RED}Some tests failed.${NC}"
        return 1
    fi
}

# Show usage
usage() {
    echo "Usage: $0 [test_names...]"
    echo ""
    echo "Available tests:"
    echo "  help     - Test help message"
    echo "  basic    - Basic execution test"
    echo "  multiple - Multiple metrics test"
    echo "  overhead - All overhead metrics"
    echo "  llm      - LLM metrics"
    echo "  json     - JSON output only"
    echo "  verbose  - Verbose output"
    echo "  limits   - Custom limits"
    echo "  reports  - Report generation"
    echo "  full     - Full benchmark (all metrics)"
    echo "  all      - Run all tests"
    echo ""
    echo "Default: Runs quick tests (help basic multiple overhead llm json verbose limits reports)"
}

if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    usage
    exit 0
fi

main "$@"
