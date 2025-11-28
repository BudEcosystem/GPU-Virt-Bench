# GPU Virtualization Performance Evaluation Tool
# Simple Makefile (alternative to CMake)

# Compilers
NVCC := nvcc
CC := gcc

# CUDA paths (adjust if needed)
CUDA_PATH ?= /usr/local/cuda
CUDA_INC := $(CUDA_PATH)/include
CUDA_LIB := $(CUDA_PATH)/lib64

# Compiler flags
CFLAGS := -Wall -Wextra -O2 -I./src -I$(CUDA_INC)
NVCCFLAGS := -O2 -I./src -I$(CUDA_INC)

# CUDA architectures (adjust based on your GPU)
# 70=V100, 75=T4, 80=A100, 86=RTX3090, 89=RTX4090, 90=H100
CUDA_ARCH := -gencode arch=compute_70,code=sm_70 \
             -gencode arch=compute_75,code=sm_75 \
             -gencode arch=compute_80,code=sm_80 \
             -gencode arch=compute_86,code=sm_86

# Libraries
LDFLAGS := -L$(CUDA_LIB) -L/usr/lib/x86_64-linux-gnu -lcudart -lnvidia-ml -lm -lpthread -ldl

# Directories
BUILD_DIR := build
SRC_DIR := src
OBJ_DIR := $(BUILD_DIR)/obj

# Source files
C_SOURCES := $(SRC_DIR)/main.c \
             $(SRC_DIR)/utils/statistics.c \
             $(SRC_DIR)/utils/reporting.c \
             $(SRC_DIR)/utils/process.c \
             $(SRC_DIR)/systems/mig_simulator.c

CU_SOURCES := $(SRC_DIR)/utils/timing.cu \
              $(SRC_DIR)/metrics/overhead.cu \
              $(SRC_DIR)/metrics/isolation.cu \
              $(SRC_DIR)/metrics/llm.cu \
              $(SRC_DIR)/metrics/bandwidth.cu \
              $(SRC_DIR)/metrics/cache.cu \
              $(SRC_DIR)/metrics/error.cu \
              $(SRC_DIR)/metrics/fragmentation.cu \
              $(SRC_DIR)/metrics/nccl.cu \
              $(SRC_DIR)/metrics/pcie.cu \
              $(SRC_DIR)/metrics/scheduling.cu \
              $(SRC_DIR)/metrics/paper_features.cu

# Object files
C_OBJECTS := $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(C_SOURCES))
CU_OBJECTS := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.cu.o,$(CU_SOURCES))
OBJECTS := $(C_OBJECTS) $(CU_OBJECTS)

# Target
TARGET := $(BUILD_DIR)/gpu-virt-bench

# Output directory
BENCHMARK_DIR := benchmarks

.PHONY: all clean install benchmark-native benchmark-hami benchmark-fcsp benchmark-all help

all: $(TARGET)

$(TARGET): $(OBJECTS)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -o $@ $^ $(LDFLAGS)
	@echo "Build complete: $(TARGET)"

# C source compilation
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c -o $@ $<

# CUDA source compilation
$(OBJ_DIR)/%.cu.o: $(SRC_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -c -o $@ $<

# Create benchmarks directory
$(BENCHMARK_DIR):
	@mkdir -p $(BENCHMARK_DIR)

# Benchmark targets
benchmark-native: $(TARGET) $(BENCHMARK_DIR)
	./$(TARGET) --system native --output $(BENCHMARK_DIR)

benchmark-hami: $(TARGET) $(BENCHMARK_DIR)
	./$(TARGET) --system hami --output $(BENCHMARK_DIR)

benchmark-fcsp: $(TARGET) $(BENCHMARK_DIR)
	./$(TARGET) --system fcsp --output $(BENCHMARK_DIR)

benchmark-all: $(TARGET) $(BENCHMARK_DIR)
	./$(TARGET) --system native --output $(BENCHMARK_DIR)
	./$(TARGET) --system hami --output $(BENCHMARK_DIR)
	./$(TARGET) --system fcsp --output $(BENCHMARK_DIR)

# Quick test
test: $(TARGET)
	./$(TARGET) --system native --iterations 10 --warmup 2 --metrics OH-001,OH-002

clean:
	rm -rf $(BUILD_DIR)

install: $(TARGET)
	install -m 755 $(TARGET) /usr/local/bin/

help:
	@echo "GPU Virtualization Performance Evaluation Tool"
	@echo ""
	@echo "Targets:"
	@echo "  all              - Build the benchmark tool (default)"
	@echo "  clean            - Remove build artifacts"
	@echo "  install          - Install to /usr/local/bin"
	@echo "  test             - Run quick test benchmark"
	@echo "  benchmark-native - Run native (no virtualization) benchmark"
	@echo "  benchmark-hami   - Run HaMi-core benchmark"
	@echo "  benchmark-fcsp   - Run FCSP benchmark"
	@echo "  benchmark-all    - Run all benchmarks"
	@echo ""
	@echo "Environment variables:"
	@echo "  CUDA_PATH        - CUDA toolkit path (default: /usr/local/cuda)"
	@echo ""
	@echo "Usage after build:"
	@echo "  ./build/gpu-virt-bench --help"
