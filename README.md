# TensAxon (`linalg`)

A lightweight, header-only C++20 linear algebra library providing fixed-size, stack-allocated `Vector` and `Matrix` types with `constexpr` support, basic matrix decompositions/operations, and a Gaussian elimination linear solver.

## Features

- **Header-only** (`INTERFACE` CMake library target `LinAlg::linalg`)
- **C++20 `constexpr` API** for compile-time and runtime linear algebra
- **Fixed-size `Vector<T, N>`**:
  - Element access (`operator[]`)
  - Arithmetic (`+=`, `-=`, `*=`, `/=`, `+`, `-`, scalar `*`)
  - `dot()`, `squared_norm()`, `norm()`
- **Fixed-size `Matrix<T, Rows, Cols>`** (row-major `std::array<T, Rows * Cols>`):
  - Element access `operator()(r, c)`
  - `identity()`, `transpose()`
  - Matrix arithmetic (`+=`, `*=`, `+`, `-`, matrix `*` matrix, matrix `*` vector)
  - `determinant()` (closed-form for 1×1, 2×2, 3×3; Gaussian elimination with partial pivoting for $N \ge 4$)
  - `inverse(eps)` returning `std::optional<Matrix<T, Rows, Cols>>` (closed-form for 1×1–3×3; Gauss-Jordan elimination with partial pivoting for $N \ge 4$)
- **Linear System Solver**:
  - `solve(A, b, eps)` solving $Ax = b$ via Gaussian elimination with partial pivoting and back-substitution, returning `std::optional<Vector<T, N>>`

## Project Structure

```text
.
├── CMakeLists.txt          # Root CMake configuration (exports LinAlg::linalg)
├── include/
│   ├── vector.hpp          # Vector<T, N> implementation
│   └── matrix.hpp          # Matrix<T, Rows, Cols> and solve() implementation
├── tests/
│   ├── CMakeLists.txt      # Unit test target & CTest registration
│   └── tests.cpp           # Unit tests for Vector, Matrix, and solve()
├── examples/               # (Reserved for usage examples)
└── src/                    # (Reserved for source files)
```

## Requirements

- CMake 3.20 or higher
- A C++20 compatible compiler (GCC, Clang, or MSVC)

## Building & Running Tests

```bash
cmake -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## Usage

### CMake Integration

After installing or adding as a subdirectory, link against `LinAlg::linalg`:

```cmake
target_link_libraries(your_target PRIVATE LinAlg::linalg)
```

### Example

```cpp
#include "vector.hpp"
#include "matrix.hpp"

int main() {
    Matrix<double, 3, 3> A{{
         2.0,  1.0, -1.0,
        -3.0, -1.0,  2.0,
        -2.0,  1.0,  2.0
    }};
    Vector<double, 3> b{{8.0, -11.0, -3.0}};

    if (auto x = solve(A, b)) {
        // x = {2.0, 3.0, -1.0}
    }
}
```
