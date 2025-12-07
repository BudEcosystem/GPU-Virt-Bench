cmake_minimum_required(VERSION 3.18)
project(gpu-virt-bench LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -O2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CUDAToolkit_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/../bud_fcsp/src/uvm)

# Source files
set(SOURCES
    src/main.c
    src/utils/statistics.c
    src/utils/process.c
    src/utils/config.c
    ${CMAKE_SOURCE_DIR}/../bud_fcsp/src/uvm/idle_detector.c
    ${CMAKE_SOURCE_DIR}/../bud_fcsp/src/uvm/pressure_monitor.c
    ${CMAKE_SOURCE_DIR}/../bud_fcsp/src/uvm/transfer_manager.c
    ${CMAKE_SOURCE_DIR}/../bud_fcsp/src/uvm/eviction_policy.c
    ${CMAKE_SOURCE_DIR}/../bud_fcsp/src/uvm/prefetch_manager.c
    ${CMAKE_SOURCE_DIR}/../bud_fcsp/src/graph/multi_graph_manager.c
)

set(CUDA_SOURCES
    src/utils/timing.cu
    src/metrics/overhead.cu
    src/metrics/isolation.cu
    src/metrics/llm.cu
    src/metrics/bandwidth.cu
    src/metrics/cache.cu
    src/metrics/pcie.cu
    src/metrics/nccl.cu
    src/metrics/scheduling.cu
    src/metrics/fragmentation.cu
    src/metrics/error.cu
    src/metrics/paper_features.cu
    src/metrics/fcsp_features.cu
    src/metrics/uvm_benchmarks.cu
    src/metrics/idle_detector_bench.cu
    src/metrics/pressure_monitor_bench.cu
    src/metrics/transfer_manager_bench.cu
    src/metrics/eviction_policy_bench.cu
    src/metrics/prefetch_manager_bench.cu
    src/metrics/multi_graph_bench.cu
    src/systems/mig_simulator.c
)

# Main executable
add_executable(gpu-virt-bench
    ${SOURCES}
    ${CUDA_SOURCES}
)

# Link libraries
target_link_libraries(gpu-virt-bench
    CUDA::cudart
    CUDA::nvml
    m
    pthread
    dl
)

# Set CUDA architecture (adjust based on your GPU)
# Common architectures:
#   70 = V100
#   75 = T4
#   80 = A100
#   86 = RTX 3090, A10
#   89 = RTX 4090, L40
#   90 = H100
set_target_properties(gpu-virt-bench PROPERTIES
    CUDA_ARCHITECTURES "70;75;80;86;89;90"
)

# Installation
install(TARGETS gpu-virt-bench
    RUNTIME DESTINATION bin
)

# Create benchmarks output directory
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/benchmarks)

# Testing support
enable_testing()

# Add test that runs a quick benchmark
add_test(
    NAME quick_benchmark_test
    COMMAND gpu-virt-bench --system native --iterations 10 --warmup 2 --overhead
)

# Custom targets
add_custom_target(benchmark-native
    COMMAND $<TARGET_FILE:gpu-virt-bench> --system native --output ${CMAKE_SOURCE_DIR}/benchmarks
    DEPENDS gpu-virt-bench
    COMMENT "Running native benchmark..."
)

add_custom_target(benchmark-hami
    COMMAND $<TARGET_FILE:gpu-virt-bench> --system hami --output ${CMAKE_SOURCE_DIR}/benchmarks
    DEPENDS gpu-virt-bench
    COMMENT "Running HaMi-core benchmark..."
)

add_custom_target(benchmark-fcsp
    COMMAND $<TARGET_FILE:gpu-virt-bench> --system fcsp --output ${CMAKE_SOURCE_DIR}/benchmarks
    DEPENDS gpu-virt-bench
    COMMENT "Running FCSP benchmark..."
)

add_custom_target(benchmark-all
    COMMAND $<TARGET_FILE:gpu-virt-bench> --system native --output ${CMAKE_SOURCE_DIR}/benchmarks
    COMMAND $<TARGET_FILE:gpu-virt-bench> --system hami --output ${CMAKE_SOURCE_DIR}/benchmarks
    COMMAND $<TARGET_FILE:gpu-virt-bench> --system fcsp --output ${CMAKE_SOURCE_DIR}/benchmarks
    DEPENDS gpu-virt-bench
    COMMENT "Running all benchmarks..."
)

# Print configuration summary
message(STATUS "")
message(STATUS "GPU Virtualization Benchmark Tool Configuration:")
message(STATUS "  CUDA Toolkit: ${CUDAToolkit_VERSION}")
message(STATUS "  CUDA Include: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "")
