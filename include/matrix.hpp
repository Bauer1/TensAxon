#pragma once
#include <cstddef>
#include <array>
#include <optional>
#include <utility>
#include "vector.hpp"

template<typename T, std::size_t Rows, std::size_t Cols>
struct Matrix {
    static_assert(Rows > 0 && Cols > 0, "Matrix dimensions must be positive");
    std::array<T, Rows * Cols> data{};

    constexpr T& operator()(std::size_t r, std::size_t c) noexcept {
        return data[r * Cols + c];
    }
    constexpr const T& operator()(std::size_t r, std::size_t c) const noexcept {
        return data[r * Cols + c];
    }

    constexpr Matrix& operator+=(const Matrix& other) noexcept {
        for (std::size_t i = 0; i < Rows * Cols; ++i) data[i] += other.data[i];
        return *this;
    }

    constexpr Matrix& operator*=(T scalar) noexcept {
        for (std::size_t i = 0; i < Rows * Cols; ++i) data[i] *= scalar;
        return *this;
    }

    static constexpr Matrix identity() noexcept {
        static_assert(Rows == Cols, "Identity is only defined for square matrices");
        Matrix res{};
        for (std::size_t i = 0; i < Rows * Cols; ++i) res(i, i) = T{1};
        return res;
    }

    constexpr Matrix<T, Cols, Rows> transpose() const noexcept {
        Matrix<T, Rows, Cols> result{};
        for (std::size_t r = 0; r < Rows; ++r) {
            for (std::size_t c = 0; c < Cols; ++c) {
                result(c, r) = (*this)(r, c);
            }
        }
        return result;
    }

    constexpr T determinant() const noexcept {
        static_assert(Rows == Cols, "Matrix must be squared");
        constexpr std::size_t N = Rows;

        if constexpr (N == 1) {
            return (*this)(0, 0);
        } else if constexpr (N == 2) {
            return (*this)(0, 0) * (*this)(1, 1) - (*this)(0, 1) * (*this)(1, 0);
        } else if constexpr (N == 3) {
            return (*this)(0, 0) * ((*this)(1, 1) * (*this)(2, 2) - (*this)(1, 2) * (*this)(2, 1))
                 - (*this)(0, 1) * ((*this)(1, 0) * (*this)(2, 2) - (*this)(1, 2) * (*this)(2, 0))
                 + (*this)(0, 2) * ((*this)(1, 0) * (*this)(2, 1) - (*this)(1, 1) * (*this)(2, 0));
        } else {
            Matrix<T, N, N> mat = *this;
            T det{1};
            int sign = 1;
            
            for (std::size_t i = 0; i < N; ++i) {
                std::size_t pivot_row = i;
                T max_val = std::abs(mat(i, i));

                for (std::size_t r = i + 1; r < N; ++r) {
                    T val = std::abs(mat(r, i));
                    if (val > max_val) {
                        max_val = val;
                        pivot_row = r;
                    }
                }

                if (max_val == T{0}) {
                    return T{0};
                }

                if (pivot_row != i) {
                    for (std::size_t c = i; c < N; ++c) {
                        std::swap(mat(i, c), mat(pivot_row, c));
                    }
                    sign = -sign;
                }

                T pivot = mat(i, i);
                for (std::size_t r = i + 1; r < N; ++r) {
                    T factor = mat(r, i) / pivot;
                    for (std::size_t c = i + 1; c < N; ++c) {
                        mat(r, c) -= factor * mat(i, c);
                    }
                }

                det *= pivot;
            }

            return det * static_cast<T>(sign);
        }
    }

    constexpr std::optional<Matrix<T, Rows, Cols>> inverse(T eps = T{1e-7}) const noexcept {
        static_assert(Rows == Cols, "Matrix must be square");
        constexpr std::size_t N = Rows;

        if constexpr (N == 1) {
            if (std::abs((*this)(0, 0)) <= eps) return std::nullopt;
            Matrix res{};
            res(0, 0) = T{1} / (*this)(0, 0);
            return res;
        } else if constexpr (N == 2) {
            T det = (*this)(0, 0) * (*this)(1, 1) -  (*this)(1, 0) * (*this)(0,1);
            if (std::abs(det) <= eps) return std::nullopt;

            T inv_det = T{1} / det;
            Matrix res{};
            res(0, 0) = (*this)(1, 1) * inv_det;
            res(0, 1) = (*this)(0, 1) * inv_det;
            res(1, 0) = (*this)(1, 0) * inv_det;
            res(1, 1) = (*this)(1, 1) * inv_det;
            return res;
        } else if constexpr (N == 3) {
            const T a00 = (*this)(0, 0), a01 = (*this)(0, 1), a02 = (*this)(0, 2);
            const T a10 = (*this)(1, 0), a11 = (*this)(1, 1), a12 = (*this)(1, 2);
            const T a20 = (*this)(2, 0), a21 = (*this)(2, 1), a22 = (*this)(2, 2);

            const T c00 =  (a11 * a22 - a12 * a21);
            const T c01 = -(a01 * a22 - a02 * a21);
            const T c02 =  (a01 * a12 - a02 * a11);

            const T c10 = -(a10 * a22 - a12 * a20);
            const T c11 =  (a00 * a22 - a02 * a20);
            const T c12 = -(a00 * a12 - a02 * a10);

            const T c20 =  (a10 * a21 - a11 * a20);
            const T c21 = -(a00 * a21 - a01 * a20);
            const T c22 =  (a00 * a11 - a01 * a10);

            T det = a00 * c00 + a01 * c10 + a02 * c20;
            if (std::abs(det) <= eps) return std::nullopt;

            T inv_det = T{1} / det;
            Matrix res{};
            res(0, 0) = c00 * inv_det; res(0, 1) = c01 * inv_det; res(0, 2) = c02 * inv_det;
            res(1, 0) = c10 * inv_det; res(1, 1) = c11 * inv_det; res(1, 2) = c12 * inv_det;
            res(2, 0) = c20 * inv_det; res(2, 1) = c21 * inv_det; res(2, 2) = c22 * inv_det;
            return res;
        } else {
            Matrix<T, N, N> A = *this;
            Matrix<T, N, N> inv = Matrix<T, N, N>::identity();

            for (std::size_t i = 0; i < N; ++i) {
                std::size_t pivot_row = i;
                T max_val = std::abs(A(i, i));

                for (std::size_t r = i + 1; r < N; ++r) {
                    T val = std::abs(A(r, i));
                    if (val > max_val) {
                        max_val = val;
                        pivot_row = r;
                    }
                }

                if (max_val <= eps) return std::nullopt;

                if (pivot_row != i) {
                    for (std::size_t c = 0; c < N; ++c) {
                        std::swap(A(i, c), A(pivot_row, c));
                        std::swap(inv(i, c), inv(pivot_row, c));
                    }
                }
                
                T pivot = A(i, i);
                T inv_pivot = T{1} / pivot;
                for (std::size_t c = 0; c < N; ++c) {
                    A(i, c) *= inv_pivot;
                    inv(i, c) *= inv_pivot;
                }

                for (std::size_t r = 0; r < N; ++r) {
                    if (r != i) {
                        T factor = A(r, i);
                        if (factor != T{0}) {
                            for (std::size_t c = 0; c < N; ++c) {
                                A(r, c) -= factor * A(i, c);
                                inv(r, c) -= factor * inv(i, c);
                            }
                        }
                    }
                }
            }

            return inv;
        }
    }
};

template<typename T, std::size_t Rows, std::size_t Cols> 
constexpr Matrix<T, Rows, Cols> operator+(Matrix<T, Rows, Cols> lhs, const Matrix<T, Rows, Cols>& rhs) noexcept {
    lhs += rhs;
    return lhs;
}

template<typename T, std::size_t Rows, std::size_t Cols> 
constexpr Matrix<T, Rows, Cols> operator-(Matrix<T, Rows, Cols> lhs, const Matrix<T, Rows, Cols>& rhs) noexcept {
    lhs -= rhs;
    return lhs;
}

template<typename T, std::size_t Rows, std::size_t K, std::size_t Cols>
constexpr Matrix<T, Rows, Cols> operator*(const Matrix<T, Rows, K>& lhs,
                                          const Matrix<T, K, Cols>& rhs) noexcept {
    Matrix<T, Rows, Cols> result{};
    for (std::size_t r = 0; r < Rows; ++r) {
        for (std::size_t k = 0; k < K; ++k) {
            T factor = lhs(r, k);
            for (std::size_t c = 0; c < Cols; ++c) {
                result(r, c) += factor * rhs(k, c);
            }
        }
    }
    return result;
}

template<typename T, std::size_t Rows, std::size_t Cols>
constexpr Vector<T, Rows> operator*(const Matrix<T, Rows, Cols>& mat,
                                    const Vector<T, Cols>& vec) noexcept {
    Vector<T, Rows> result{};
    for (std::size_t r = 0; r < Rows; ++r) {
        T sum{0};
        for (std::size_t c = 0; c < Cols; ++c) {
            sum += mat(r, c) * vec[c];
        }
        result[r] = sum;
    }
    return result;
}

template<typename T, std::size_t N>
constexpr std::optional<Vector<T, N>> solve(
    Matrix<T, N, N> A,
    Vector<T, N> b,
    T eps = T{1e-7}
) noexcept {
    for (std::size_t i = 0; i < N; ++i) {
        std::size_t pivot_row = i;
        T max_val = std::abs(A(i, i));

        for (std::size_t r = i + 1; r < N; ++r) {
            T val = std::abs(A(r, i));
            if (val > max_val) {
                max_val = val;
                pivot_row = r;
            }
        }

        if (max_val <= eps) return std::nullopt;

        if (pivot_row != i) {
            for (std::size_t c = i; c < N; ++c) {
                std::swap(A(i, c), A(pivot_row, c));
            }
            std::swap(b[i], b[pivot_row]);
        }

        T pivot = A(i, i);
        for (std::size_t r = i + 1; r < N; ++r) {
            T factor = A(r, i) / pivot;

            for (std::size_t c = i + 1; c < N; ++c) {
                A(r, c) -= factor * A(i, c);
            }
            b[r] -= factor * b[i];
        }
    }

    Vector<T, N> x{};
    for (int i = static_cast<int>(N) - 1; i >= 0; --i) {
        T sum = b[i];
        for (std::size_t j = static_cast<std::size_t>(i) + 1; j < N; ++j) {
            sum -= A(i, j) * x[j];
        }
        x[i] = sum / A(i, i);
    }
    return x;
}