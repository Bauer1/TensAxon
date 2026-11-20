# TensAxon

[![CI](https://github.com/Bauer1/TensAxon/actions/workflows/ci.yml/badge.svg)](https://github.com/Bauer1/TensAxon/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)

A lightweight, **header-only C++20** library for linear algebra and N-dimensional tensors. All types are fixed-size and stack-allocated, their shapes are part of the type (so dimension mismatches are **compile errors**), and the whole API is `constexpr`.

```cpp
#include <tensaxon.hpp>
using namespace tensaxon;

Mat3<double> A{{ 2, 1, -1,   -3, -1, 2,   -2, 1, 2 }};
Vec3<double> b{{ 8, -11, -3 }};

auto x = solve(A, b);                              // std::optional -> {2, 3, -1}
auto T = outer(to_tensor(b), to_tensor(b));        // Tensor<double, 3, 3>
auto y = contract<1, 0>(to_tensor(A), to_tensor(b)); // == A * b
```

## Features

### `Vector<T, N>` — [`vector.hpp`](include/vector.hpp)
- `+`, `-`, unary `-`, scalar `*` / `/`, compound assignment, `==`
- `dot()`, `squared_norm()`, `norm()`, `normalized()`, `cross()` (3D only)
- Factories: `zeros()`, `ones()`, `filled(v)`

### `Matrix<T, Rows, Cols>` — [`matrix.hpp`](include/matrix.hpp)
- Row-major storage, `m(r, c)` access, `row(i)`, `col(j)`, `swap_rows()`
- `+`, `-`, unary `-`, scalar `*` / `/`, `==`, matrix × matrix, matrix × vector
- `identity()`, `transpose()` (any shape), `trace()`
- `determinant()` — closed form for N ≤ 3, Gaussian elimination with partial pivoting for N ≥ 4
- `inverse(eps)` → `std::optional<Matrix>` — closed form for N ≤ 3, Gauss–Jordan for N ≥ 4
- `solve(A, b, eps)` → `std::optional<Vector>` — solves *Ax = b* with partial pivoting
- Aliases: `Vec2/3/4<T>`, `Mat2/3/4<T>`

### `Tensor<T, Dims...>` — [`tensor.hpp`](include/tensor.hpp)
- Any rank (including rank-0 scalars): `Tensor<float, 2, 3, 4> t{}; t(1, 2, 3) = 1.0f;`
- Compile-time `rank`, `size`, `shape`, `strides`; `offset()` / `unravel()` index helpers
- Element-wise `+`, `-`, scalar `*` / `/`, `hadamard()`, `==`
- Reductions: `sum()`, `squared_norm()`, `norm()` (Frobenius)
- Shape ops: `reshape<...>()`, `flatten()`, `permute<axes...>()`, `transpose()` (rank 2)
- `outer(a, b)` — tensor product, shapes concatenate
- `contract<AxisA, AxisB>(a, b)` — sums over one shared axis (generalizes dot, mat-vec and mat-mat products)
- Interop (identical row-major layout): `to_tensor(Vector/Matrix)`, `to_vector()`, `to_matrix()`

## Project Structure

```text
.
├── CMakeLists.txt
├── include/
│   ├── tensaxon.hpp            # umbrella header
│   ├── vector.hpp
│   ├── matrix.hpp              # Matrix + solve()
│   ├── tensor.hpp
│   └── detail/
│       └── math.hpp            # constexpr abs/sqrt, concepts, tolerances
├── tests/                      # unit tests (CTest)
└── examples/basic.cpp
```

## Building

Requires CMake ≥ 3.20 and a C++20 compiler (GCC 10+, Clang 13+, MSVC 19.29+).

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
./build/examples/tensaxon_example
```

Options (default `ON` when TensAxon is the top-level project):

| Option | Description |
| --- | --- |
| `TENSAXON_BUILD_TESTS` | Build the unit tests |
| `TENSAXON_BUILD_EXAMPLES` | Build the examples |
| `TENSAXON_ENABLE_SANITIZERS` | Enable AddressSanitizer and UndefinedBehaviorSanitizer |

## Using in Your Project

**As a subdirectory / FetchContent:**

```cmake
include(FetchContent)
FetchContent_Declare(
    TensAxon
    GIT_REPOSITORY https://github.com/Bauer1/TensAxon.git
    GIT_TAG        v0.2.0
)
FetchContent_MakeAvailable(TensAxon)

target_link_libraries(your_target PRIVATE TensAxon::tensaxon)
```

**After `cmake --install`:**

```cmake
find_package(TensAxon 0.2 REQUIRED)
target_link_libraries(your_target PRIVATE TensAxon::tensaxon)
```

## Notes

- `inverse()`, `solve()`, `norm()` and `determinant()` for N ≥ 4 require a floating-point element type; this is enforced at compile time.
- Singularity is detected with an absolute pivot tolerance (`eps`, default `1e-7`). Scale your data or pass a custom `eps` if needed.

## License

See [LICENSE](LICENSE).
