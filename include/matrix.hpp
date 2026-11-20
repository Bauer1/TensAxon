#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <optional>
#include <utility>

#include "detail/math.hpp"
#include "vector.hpp"

namespace tensaxon {

/// Fixed-size, stack-allocated, row-major matrix.
///
/// `Matrix` is an aggregate; elements are listed row by row:
/// @code
///   tensaxon::Matrix<double, 2, 2> m{{1.0, 2.0,
///                                     3.0, 4.0}};
/// @endcode
template <Arithmetic T, std::size_t Rows, std::size_t Cols>
struct Matrix {
    static_assert(Rows > 0 && Cols > 0, "Matrix dimensions must be positive");

    using value_type = T;
    static constexpr std::size_t rows = Rows;
    static constexpr std::size_t cols = Cols;
    static constexpr std::size_t size = Rows * Cols;
    static constexpr bool is_square = Rows == Cols;

    std::array<T, Rows * Cols> data{};

    // ----- element access ---------------------------------------------------
    [[nodiscard]] constexpr T& operator()(std::size_t r, std::size_t c) noexcept {
        assert(r < Rows && c < Cols && "Matrix index out of range");
        return data[r * Cols + c];
    }
    [[nodiscard]] constexpr const T& operator()(std::size_t r, std::size_t c) const noexcept {
        assert(r < Rows && c < Cols && "Matrix index out of range");
        return data[r * Cols + c];
    }

    [[nodiscard]] constexpr Vector<T, Cols> row(std::size_t r) const noexcept {
        Vector<T, Cols> v{};
        for (std::size_t c = 0; c < Cols; ++c) v[c] = (*this)(r, c);
        return v;
    }
    [[nodiscard]] constexpr Vector<T, Rows> col(std::size_t c) const noexcept {
        Vector<T, Rows> v{};
        for (std::size_t r = 0; r < Rows; ++r) v[r] = (*this)(r, c);
        return v;
    }

    constexpr void swap_rows(std::size_t a, std::size_t b) noexcept {
        if (a == b) return;
        for (std::size_t c = 0; c < Cols; ++c) std::swap((*this)(a, c), (*this)(b, c));
    }

    // ----- factories --------------------------------------------------------
    [[nodiscard]] static constexpr Matrix zeros() noexcept { return Matrix{}; }

    [[nodiscard]] static constexpr Matrix filled(T value) noexcept {
        Matrix m;
        m.data.fill(value);
        return m;
    }

    [[nodiscard]] static constexpr Matrix identity() noexcept
        requires(Rows == Cols)
    {
        Matrix res{};
        for (std::size_t i = 0; i < Rows; ++i) res(i, i) = T{1};
        return res;
    }

    // ----- compound assignment ----------------------------------------------
    constexpr Matrix& operator+=(const Matrix& other) noexcept {
        for (std::size_t i = 0; i < size; ++i) data[i] += other.data[i];
        return *this;
    }
    constexpr Matrix& operator-=(const Matrix& other) noexcept {
        for (std::size_t i = 0; i < size; ++i) data[i] -= other.data[i];
        return *this;
    }
    constexpr Matrix& operator*=(T scalar) noexcept {
        for (auto& x : data) x *= scalar;
        return *this;
    }
    constexpr Matrix& operator/=(T scalar) noexcept {
        assert(scalar != T{0} && "Division by zero");
        for (auto& x : data) x /= scalar;
        return *this;
    }

    [[nodiscard]] friend constexpr bool operator==(const Matrix&, const Matrix&) = default;

    // ----- structural operations --------------------------------------------
    [[nodiscard]] constexpr Matrix<T, Cols, Rows> transpose() const noexcept {
        Matrix<T, Cols, Rows> result{};
        for (std::size_t r = 0; r < Rows; ++r)
            for (std::size_t c = 0; c < Cols; ++c) result(c, r) = (*this)(r, c);
        return result;
    }

    [[nodiscard]] constexpr T trace() const noexcept
        requires(Rows == Cols)
    {
        T sum{0};
        for (std::size_t i = 0; i < Rows; ++i) sum += (*this)(i, i);
        return sum;
    }

    // ----- determinant ------------------------------------------------------
    /// Closed form for N <= 3, Gaussian elimination with partial pivoting
    /// otherwise (the latter requires a floating-point element type).
    [[nodiscard]] constexpr T determinant() const noexcept
        requires(Rows == Cols)
    {
        constexpr std::size_t N = Rows;
        const Matrix& m = *this;

        if constexpr (N == 1) {
            return m(0, 0);
        } else if constexpr (N == 2) {
            return m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0);
        } else if constexpr (N == 3) {
            return m(0, 0) * (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1))
                 - m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0))
                 + m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0));
        } else {
            static_assert(std::floating_point<T>,
                          "determinant() for N >= 4 requires a floating-point type");
            Matrix a = m;
            T det{1};
            for (std::size_t i = 0; i < N; ++i) {
                const std::size_t p = a.pivot_row(i);
                if (a(p, i) == T{0}) return T{0};
                if (p != i) {
                    a.swap_rows(i, p);
                    det = -det;
                }
                const T pivot = a(i, i);
                for (std::size_t r = i + 1; r < N; ++r) {
                    const T factor = a(r, i) / pivot;
                    for (std::size_t c = i + 1; c < N; ++c) a(r, c) -= factor * a(i, c);
                }
                det *= pivot;
            }
            return det;
        }
    }

    // ----- inverse ----------------------------------------------------------
    /// Returns std::nullopt if the matrix is singular (|pivot| <= eps).
    /// Closed form for N <= 3, Gauss-Jordan elimination otherwise.
    [[nodiscard]] constexpr std::optional<Matrix> inverse(T eps = detail::default_eps<T>) const noexcept
        requires(Rows == Cols && std::floating_point<T>)
    {
        constexpr std::size_t N = Rows;
        const Matrix& m = *this;

        if constexpr (N == 1) {
            if (detail::abs(m(0, 0)) <= eps) return std::nullopt;
            return Matrix{{T{1} / m(0, 0)}};
        } else if constexpr (N == 2) {
            const T det = determinant();
            if (detail::abs(det) <= eps) return std::nullopt;
            const T inv_det = T{1} / det;
            return Matrix{{ m(1, 1) * inv_det, -m(0, 1) * inv_det,
                           -m(1, 0) * inv_det,  m(0, 0) * inv_det}};
        } else if constexpr (N == 3) {
            const T a00 = m(0, 0), a01 = m(0, 1), a02 = m(0, 2);
            const T a10 = m(1, 0), a11 = m(1, 1), a12 = m(1, 2);
            const T a20 = m(2, 0), a21 = m(2, 1), a22 = m(2, 2);

            // Adjugate (transposed cofactor matrix).
            const Matrix adj{{
                 (a11 * a22 - a12 * a21), -(a01 * a22 - a02 * a21),  (a01 * a12 - a02 * a11),
                -(a10 * a22 - a12 * a20),  (a00 * a22 - a02 * a20), -(a00 * a12 - a02 * a10),
                 (a10 * a21 - a11 * a20), -(a00 * a21 - a01 * a20),  (a00 * a11 - a01 * a10),
            }};

            const T det = a00 * adj(0, 0) + a01 * adj(1, 0) + a02 * adj(2, 0);
            if (detail::abs(det) <= eps) return std::nullopt;

            Matrix res = adj;
            res *= T{1} / det;
            return res;
        } else {
            Matrix a = m;
            Matrix inv = identity();
            for (std::size_t i = 0; i < N; ++i) {
                const std::size_t p = a.pivot_row(i);
                if (detail::abs(a(p, i)) <= eps) return std::nullopt;
                a.swap_rows(i, p);
                inv.swap_rows(i, p);

                const T inv_pivot = T{1} / a(i, i);
                for (std::size_t c = 0; c < N; ++c) {
                    a(i, c) *= inv_pivot;
                    inv(i, c) *= inv_pivot;
                }
                for (std::size_t r = 0; r < N; ++r) {
                    if (r == i) continue;
                    const T factor = a(r, i);
                    if (factor == T{0}) continue;
                    for (std::size_t c = 0; c < N; ++c) {
                        a(r, c) -= factor * a(i, c);
                        inv(r, c) -= factor * inv(i, c);
                    }
                }
            }
            return inv;
        }
    }

    /// Index of the row (>= col) with the largest |value| in column `col`.
    [[nodiscard]] constexpr std::size_t pivot_row(std::size_t col) const noexcept {
        std::size_t best = col;
        T best_val = detail::abs((*this)(col, col));
        for (std::size_t r = col + 1; r < Rows; ++r) {
            const T v = detail::abs((*this)(r, col));
            if (v > best_val) {
                best_val = v;
                best = r;
            }
        }
        return best;
    }
};

// ----- free operators ---------------------------------------------------------
template <typename T, std::size_t R, std::size_t C>
[[nodiscard]] constexpr Matrix<T, R, C> operator+(Matrix<T, R, C> lhs, const Matrix<T, R, C>& rhs) noexcept {
    return lhs += rhs;
}

template <typename T, std::size_t R, std::size_t C>
[[nodiscard]] constexpr Matrix<T, R, C> operator-(Matrix<T, R, C> lhs, const Matrix<T, R, C>& rhs) noexcept {
    return lhs -= rhs;
}

template <typename T, std::size_t R, std::size_t C>
[[nodiscard]] constexpr Matrix<T, R, C> operator-(Matrix<T, R, C> m) noexcept {
    for (auto& x : m.data) x = -x;
    return m;
}

template <typename T, std::size_t R, std::size_t C>
[[nodiscard]] constexpr Matrix<T, R, C> operator*(Matrix<T, R, C> m, T scalar) noexcept {
    return m *= scalar;
}

template <typename T, std::size_t R, std::size_t C>
[[nodiscard]] constexpr Matrix<T, R, C> operator*(T scalar, Matrix<T, R, C> m) noexcept {
    return m *= scalar;
}

template <typename T, std::size_t R, std::size_t C>
[[nodiscard]] constexpr Matrix<T, R, C> operator/(Matrix<T, R, C> m, T scalar) noexcept {
    return m /= scalar;
}

/// Matrix product: (R x K) * (K x C) -> (R x C). Uses cache-friendly i-k-j order.
template <typename T, std::size_t R, std::size_t K, std::size_t C>
[[nodiscard]] constexpr Matrix<T, R, C> operator*(const Matrix<T, R, K>& lhs,
                                                  const Matrix<T, K, C>& rhs) noexcept {
    Matrix<T, R, C> result{};
    for (std::size_t r = 0; r < R; ++r)
        for (std::size_t k = 0; k < K; ++k) {
            const T factor = lhs(r, k);
            for (std::size_t c = 0; c < C; ++c) result(r, c) += factor * rhs(k, c);
        }
    return result;
}

/// Matrix-vector product: (R x C) * (C) -> (R).
template <typename T, std::size_t R, std::size_t C>
[[nodiscard]] constexpr Vector<T, R> operator*(const Matrix<T, R, C>& m,
                                               const Vector<T, C>& v) noexcept {
    Vector<T, R> result{};
    for (std::size_t r = 0; r < R; ++r) {
        T sum{0};
        for (std::size_t c = 0; c < C; ++c) sum += m(r, c) * v[c];
        result[r] = sum;
    }
    return result;
}

// ----- linear solver ----------------------------------------------------------
/// Solves A x = b via Gaussian elimination with partial pivoting and
/// back-substitution. Returns std::nullopt if A is singular (|pivot| <= eps).
template <std::floating_point T, std::size_t N>
[[nodiscard]] constexpr std::optional<Vector<T, N>> solve(Matrix<T, N, N> A, Vector<T, N> b,
                                                          T eps = detail::default_eps<T>) noexcept {
    for (std::size_t i = 0; i < N; ++i) {
        const std::size_t p = A.pivot_row(i);
        if (detail::abs(A(p, i)) <= eps) return std::nullopt;
        if (p != i) {
            A.swap_rows(i, p);
            std::swap(b[i], b[p]);
        }
        const T pivot = A(i, i);
        for (std::size_t r = i + 1; r < N; ++r) {
            const T factor = A(r, i) / pivot;
            for (std::size_t c = i + 1; c < N; ++c) A(r, c) -= factor * A(i, c);
            b[r] -= factor * b[i];
        }
    }

    Vector<T, N> x{};
    for (std::size_t i = N; i-- > 0;) {
        T sum = b[i];
        for (std::size_t j = i + 1; j < N; ++j) sum -= A(i, j) * x[j];
        x[i] = sum / A(i, i);
    }
    return x;
}

// ----- aliases ----------------------------------------------------------------
template <typename T> using Vec2 = Vector<T, 2>;
template <typename T> using Vec3 = Vector<T, 3>;
template <typename T> using Vec4 = Vector<T, 4>;
template <typename T> using Mat2 = Matrix<T, 2, 2>;
template <typename T> using Mat3 = Matrix<T, 3, 3>;
template <typename T> using Mat4 = Matrix<T, 4, 4>;

}  // namespace tensaxon
