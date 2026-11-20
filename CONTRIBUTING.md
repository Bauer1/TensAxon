# Contributing to TensAxon

Thank you for your interest in contributing to **TensAxon**!

## Overview

TensAxon is a header-only, `constexpr` C++20 library for linear algebra and N-dimensional tensors. When contributing, please keep these core design goals in mind:
- **Zero external dependencies**: Uses only the C++20 standard library.
- **Header-only & `constexpr`-first**: Operations should be evaluable at compile time whenever possible.
- **Compile-time safety**: Dimensions and element type constraints are checked at compile time via templates, `static_assert`, and C++20 concepts.

## Building and Running Tests Locally

Make sure you have CMake (>= 3.20) and a C++20-compatible compiler (GCC 10+, Clang 13+, or MSVC 19.29+).

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

To run with AddressSanitizer / UndefinedBehaviorSanitizer and check formatting / static analysis:

```bash
cmake -S . -B build-san -DCMAKE_BUILD_TYPE=Debug -DTENSAXON_ENABLE_SANITIZERS=ON
cmake --build build-san
ctest --test-dir build-san --output-on-failure

find include tests examples -type f \( -name '*.hpp' -o -name '*.cpp' \) -exec clang-format --dry-run --Werror {} +
clang-tidy -p build tests/tests.cpp examples/basic.cpp
```

## Submitting Changes

1. **Fork** the repository and create a new branch from `main`.
2. **Make your changes** in `include/`.
3. **Add or update unit tests** in `tests/` (including `static_assert` tests for `constexpr` evaluation).
4. **Verify** that all tests and examples build cleanly without warnings.
5. **Open a Pull Request** with a clear description of the motivation and changes.

## Reporting Issues

If you encounter a bug or compiler compatibility issue, please open a GitHub Issue and include:
- Your compiler name and version (e.g., `GCC 13.2`, `Clang 17`, `MSVC 19.38`)
- Your operating system
- A minimal reproducible code snippet

## License

By contributing to TensAxon, you agree that your contributions will be licensed under the [MIT License](LICENSE).
