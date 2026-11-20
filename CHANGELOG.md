# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-11-20

### Added
- `Tensor<T, Dims...>` supporting arbitrary-rank (including rank-0) fixed-size tensors (`include/tensor.hpp`).
- Tensor operations: element-wise arithmetic, `hadamard()`, `sum()`, `squared_norm()`, `norm()`, `reshape()`, `flatten()`, `permute()`, `transpose()`, `outer()`, and `contract()`.
- Interoperability helpers between `Vector`, `Matrix`, and `Tensor`: `to_tensor()`, `to_vector()`, and `to_matrix()`.
- General $N \times N$ `determinant()` and `inverse()` via Gaussian elimination / Gauss–Jordan with partial pivoting, plus `solve(A, b)` linear solver.
- Single umbrella header `include/tensaxon.hpp` and flattened `include/` directory layout.
- CMake install & package configuration (`find_package(TensAxon)` and `FetchContent` support).
- Comprehensive `constexpr` and runtime unit test suite (`tests/test_tensaxon.cpp`).
- GitHub Actions CI across Linux (GCC, Clang), macOS, and Windows (MSVC).

## [0.1.0] - 2026-11-20

### Added
- Initial release with fixed-size `Vector<T, N>` and `Matrix<T, Rows, Cols>` in C++20.
