#pragma once
// Internal helpers shared by Vector, Matrix and Tensor.
//
// std::abs / std::sqrt are not constexpr until C++23, so the library ships
// its own versions that work both at compile time and at run time.

#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <type_traits>

namespace tensaxon {

/// Element types supported by the library (integers and floating point).
template <typename T>
concept Arithmetic = std::is_arithmetic_v<T> && !std::is_same_v<T, bool>;

namespace detail {

template <typename T>
[[nodiscard]] constexpr T abs(T x) noexcept {
    return x < T{0} ? -x : x;
}

template <std::floating_point T>
[[nodiscard]] constexpr T sqrt(T x) noexcept {
    if (std::is_constant_evaluated()) {
        // Newton-Raphson iteration (compile-time path).
        if (x < T{0} || x != x) return std::numeric_limits<T>::quiet_NaN();
        if (x == T{0} || x == std::numeric_limits<T>::infinity()) return x;
        T curr = x > T{1} ? x : T{1};
        T prev = T{0};
        while (curr != prev) {
            prev = curr;
            curr = T{0.5} * (curr + x / curr);
            if (curr >= prev) break;  // converged (monotonically decreasing)
        }
        return prev < curr ? prev : curr;
    }
    return std::sqrt(x);
}

/// Default tolerance used to detect singular (near-zero) pivots.
template <std::floating_point T>
inline constexpr T default_eps = static_cast<T>(1e-7);

}  // namespace detail
}  // namespace tensaxon
