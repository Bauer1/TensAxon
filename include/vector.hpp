#pragma once
#include <array>
#include <cstddef>
#include <cmath>
#include <cassert>

template<typename T, std::size_t N>
struct Vector {
    static_assert(N > 0, "Vector dimension must be positive");
    std::array<T, N> data{};
    constexpr T& operator[](std::size_t i) noexcept { return data[i]; }
    constexpr const T& operator[](std::size_t i) const noexcept { return data[i]; }

    constexpr Vector& operator+=(const Vector& other) noexcept {
        for (std::size_t i = 0; i < N; ++i) data[i] += other.data[i];
        return *this;
    }

    constexpr Vector& operator-=(const Vector& other) noexcept {
        for (std::size_t i = 0; i < N; ++i) data[i] -= other.data[i];
        return *this;
    }

    constexpr Vector& operator*=(T scalar) noexcept {
        for (std::size_t i = 0; i < N; ++i) data[i] *= scalar;
        return *this;
    }

    constexpr Vector& operator/=(T scalar) noexcept {
        assert(scalar != T{0} && "Division by zero");
        for (std::size_t i = 0; i < N; ++i) data[i] /= scalar;
        return *this;
    }

    constexpr T dot(const Vector& other) const noexcept {
        T sum{0};
        for (std::size_t i = 0; i < N; ++i) sum += data[i] * other.data[i];
        return sum;
    }

    constexpr T squared_norm() const noexcept {
        return this->dot(*this);
    }

    constexpr T norm() const noexcept {
        return std::sqrt(squared_norm());
    }
};
    
template<typename T, std::size_t N>
constexpr Vector<T, N> operator+(Vector<T, N> lhs, const Vector<T, N> rhs) noexcept {
    lhs += rhs;
    return lhs;
}

template<typename T, std::size_t N>
constexpr Vector<T, N> operator-(Vector<T, N> lhs, const Vector<T, N> rhs) noexcept {
    lhs -= rhs;
    return lhs;
}

template<typename T, std::size_t N>
constexpr Vector<T, N> operator*(T scalar, Vector<T, N> v) noexcept {
    v *= scalar;
    return v;
}

template <typename T, std::size_t N>
constexpr Vector<T, N> operator*(Vector<T, N> v, T scalar) noexcept {
    v *= scalar;
    return v;
}

