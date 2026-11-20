#pragma once

#include <array>
#include <cassert>
#include <cstddef>

#include "detail/math.hpp"

namespace tensaxon {

/// Fixed-size, stack-allocated mathematical vector.
///
/// `Vector` is an aggregate, so it can be brace-initialised directly:
/// @code
///   tensaxon::Vector<float, 3> v{{1.0f, 2.0f, 3.0f}};
/// @endcode
template <Arithmetic T, std::size_t N>
struct Vector {
    static_assert(N > 0, "Vector dimension must be positive");

    using value_type = T;
    static constexpr std::size_t size = N;

    std::array<T, N> data{};

    // ----- element access ---------------------------------------------------
    [[nodiscard]] constexpr T& operator[](std::size_t i) noexcept {
        assert(i < N && "Vector index out of range");
        return data[i];
    }
    [[nodiscard]] constexpr const T& operator[](std::size_t i) const noexcept {
        assert(i < N && "Vector index out of range");
        return data[i];
    }

    [[nodiscard]] constexpr auto begin() noexcept { return data.begin(); }
    [[nodiscard]] constexpr auto end() noexcept { return data.end(); }
    [[nodiscard]] constexpr auto begin() const noexcept { return data.begin(); }
    [[nodiscard]] constexpr auto end() const noexcept { return data.end(); }

    // ----- factories --------------------------------------------------------
    [[nodiscard]] static constexpr Vector filled(T value) noexcept {
        Vector v;
        v.data.fill(value);
        return v;
    }
    [[nodiscard]] static constexpr Vector zeros() noexcept { return Vector{}; }
    [[nodiscard]] static constexpr Vector ones() noexcept { return filled(T{1}); }

    // ----- compound assignment ----------------------------------------------
    constexpr Vector& operator+=(const Vector& other) noexcept {
        for (std::size_t i = 0; i < N; ++i) data[i] += other.data[i];
        return *this;
    }
    constexpr Vector& operator-=(const Vector& other) noexcept {
        for (std::size_t i = 0; i < N; ++i) data[i] -= other.data[i];
        return *this;
    }
    constexpr Vector& operator*=(T scalar) noexcept {
        for (auto& x : data) x *= scalar;
        return *this;
    }
    constexpr Vector& operator/=(T scalar) noexcept {
        assert(scalar != T{0} && "Division by zero");
        for (auto& x : data) x /= scalar;
        return *this;
    }

    // ----- products & norms -------------------------------------------------
    [[nodiscard]] constexpr T dot(const Vector& other) const noexcept {
        T sum{0};
        for (std::size_t i = 0; i < N; ++i) sum += data[i] * other.data[i];
        return sum;
    }

    [[nodiscard]] constexpr T squared_norm() const noexcept { return dot(*this); }

    [[nodiscard]] constexpr T norm() const noexcept
        requires std::floating_point<T>
    {
        return detail::sqrt(squared_norm());
    }

    /// Returns a unit vector in the same direction. Asserts on a zero vector.
    [[nodiscard]] constexpr Vector normalized() const noexcept
        requires std::floating_point<T>
    {
        Vector v = *this;
        v /= norm();
        return v;
    }

    /// 3D cross product.
    [[nodiscard]] constexpr Vector cross(const Vector& o) const noexcept
        requires(N == 3)
    {
        return Vector{{data[1] * o.data[2] - data[2] * o.data[1],
                       data[2] * o.data[0] - data[0] * o.data[2],
                       data[0] * o.data[1] - data[1] * o.data[0]}};
    }

    [[nodiscard]] friend constexpr bool operator==(const Vector&, const Vector&) = default;
};

// ----- free operators ---------------------------------------------------------
template <typename T, std::size_t N>
[[nodiscard]] constexpr Vector<T, N> operator+(Vector<T, N> lhs, const Vector<T, N>& rhs) noexcept {
    return lhs += rhs;
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr Vector<T, N> operator-(Vector<T, N> lhs, const Vector<T, N>& rhs) noexcept {
    return lhs -= rhs;
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr Vector<T, N> operator-(Vector<T, N> v) noexcept {
    for (auto& x : v.data) x = -x;
    return v;
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr Vector<T, N> operator*(Vector<T, N> v, T scalar) noexcept {
    return v *= scalar;
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr Vector<T, N> operator*(T scalar, Vector<T, N> v) noexcept {
    return v *= scalar;
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr Vector<T, N> operator/(Vector<T, N> v, T scalar) noexcept {
    return v /= scalar;
}

}  // namespace tensaxon
