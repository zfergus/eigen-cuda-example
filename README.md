# Eigen CUDA Example

[![Build](https://github.com/zfergus/eigen-cuda-example/actions/workflows/continuous.yml/badge.svg)](https://github.com/zfergus/eigen-cuda-example/actions/workflows/continuous.yml)

Example project using Eigen with CUDA.

## Dependencies

All dependencies are downloaded and built automatically by CMake except for CUDA.

## Build

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

## Running Benchmarks

```bash
cd build
./tests/eigen_cuda_example_tests '[!benchmark]'
```
