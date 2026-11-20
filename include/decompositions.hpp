#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <limits>
#include <optional>
#include <utility>

#include "detail/math.hpp"
#include "matrix.hpp"
#include "vector.hpp"

namespace tensaxon {

// =============================================================================
// LU Decomposition (with partial pivoting): P * A = L * U
// =============================================================================

/// Result of an LU decomposition with partial pivoting: `P * A = L * U`.
template <std::floating_point T, std::size_t Rows, std::size_t Cols>
struct LUResult {
    static constexpr std::size_t K = Rows < Cols ? Rows : Cols;

    Matrix<T, Rows, K> L{};
    Matrix<T, K, Cols> U{};
    Matrix<T, Rows, Rows> P{};
    std::array<std::size_t, Rows> perm{};
    int sign{1};

    /// Determinant of A (square matrices only).
    [[nodiscard]] constexpr T determinant() const noexcept
        requires(Rows == Cols)
    {
        T det = static_cast<T>(sign);
        for (std::size_t i = 0; i < Rows; ++i) det *= U(i, i);
        return det;
    }

    /// Returns true if any diagonal entry of U has |U(i, i)| <= eps.
    [[nodiscard]] constexpr bool is_singular(T eps = detail::default_eps<T>) const noexcept
        requires(Rows == Cols)
    {
        for (std::size_t i = 0; i < Rows; ++i) {
            if (detail::abs(U(i, i)) <= eps) return true;
        }
        return false;
    }

    /// Solves A * x = b using the precomputed LU factors.
    /// Returns std::nullopt if A is singular (|U(i, i)| <= eps).
    [[nodiscard]] constexpr std::optional<Vector<T, Cols>> solve(
        const Vector<T, Rows>& b, T eps = detail::default_eps<T>) const noexcept
        requires(Rows == Cols)
    {
        Vector<T, Rows> y{};
        for (std::size_t i = 0; i < Rows; ++i) {
            T sum = b[perm[i]];
            for (std::size_t j = 0; j < i; ++j) sum -= L(i, j) * y[j];
            y[i] = sum;
        }

        Vector<T, Cols> x{};
        for (std::size_t i = Rows; i-- > 0;) {
            if (detail::abs(U(i, i)) <= eps) return std::nullopt;
            T sum = y[i];
            for (std::size_t j = i + 1; j < Cols; ++j) sum -= U(i, j) * x[j];
            x[i] = sum / U(i, i);
        }
        return x;
    }
};

/// Computes the LU decomposition with partial pivoting of `A`: `P * A = L * U`.
template <std::floating_point T, std::size_t Rows, std::size_t Cols>
[[nodiscard]] constexpr LUResult<T, Rows, Cols> lu(const Matrix<T, Rows, Cols>& A) noexcept {
    constexpr std::size_t K = Rows < Cols ? Rows : Cols;

    Matrix<T, Rows, Cols> work = A;
    Matrix<T, Rows, K> L{};
    std::array<std::size_t, Rows> perm{};
    for (std::size_t i = 0; i < Rows; ++i) perm[i] = i;
    int sign = 1;

    for (std::size_t i = 0; i < K; ++i) {
        std::size_t p = i;
        T best = detail::abs(work(i, i));
        for (std::size_t r = i + 1; r < Rows; ++r) {
            const T v = detail::abs(work(r, i));
            if (v > best) {
                best = v;
                p = r;
            }
        }

        if (p != i) {
            work.swap_rows(i, p);
            for (std::size_t c = 0; c < i; ++c) std::swap(L(i, c), L(p, c));
            std::swap(perm[i], perm[p]);
            sign = -sign;
        }

        L(i, i) = T{1};
        const T pivot = work(i, i);
        if (pivot != T{0}) {
            for (std::size_t r = i + 1; r < Rows; ++r) {
                const T factor = work(r, i) / pivot;
                L(r, i) = factor;
                work(r, i) = T{0};
                for (std::size_t c = i + 1; c < Cols; ++c) work(r, c) -= factor * work(i, c);
            }
        }
    }

    Matrix<T, K, Cols> U{};
    for (std::size_t r = 0; r < K; ++r)
        for (std::size_t c = 0; c < Cols; ++c) U(r, c) = work(r, c);

    Matrix<T, Rows, Rows> P{};
    for (std::size_t r = 0; r < Rows; ++r) P(r, perm[r]) = T{1};

    return {L, U, P, perm, sign};
}

// =============================================================================
// Singular Value Decomposition (One-Sided Jacobi): A = U * Sigma * V^T
// =============================================================================

/// Result of a full Singular Value Decomposition: `A = U * sigma_matrix() * V^T`.
/// Singular values `S` are non-negative and sorted in descending order.
template <std::floating_point T, std::size_t Rows, std::size_t Cols>
struct SVDResult {
    static constexpr std::size_t K = Rows < Cols ? Rows : Cols;

    Matrix<T, Rows, Rows> U{};
    Vector<T, K> S{};
    Matrix<T, Cols, Cols> V{};

    /// Constructs the Rows x Cols diagonal matrix Sigma from S.
    [[nodiscard]] constexpr Matrix<T, Rows, Cols> sigma_matrix() const noexcept {
        Matrix<T, Rows, Cols> sigma{};
        for (std::size_t i = 0; i < K; ++i) sigma(i, i) = S[i];
        return sigma;
    }

    /// Reconstructs A = U * Sigma * V^T.
    [[nodiscard]] constexpr Matrix<T, Rows, Cols> reconstruct() const noexcept {
        return U * sigma_matrix() * V.transpose();
    }

    /// Numerical rank: number of singular values strictly greater than `eps`.
    [[nodiscard]] constexpr std::size_t rank(T eps = detail::default_eps<T>) const noexcept {
        std::size_t r = 0;
        for (std::size_t i = 0; i < K; ++i) {
            if (S[i] > eps) ++r;
        }
        return r;
    }

    /// Moore-Penrose pseudoinverse A^+ = V * Sigma^+ * U^T.
    [[nodiscard]] constexpr Matrix<T, Cols, Rows> pseudoinverse(
        T eps = detail::default_eps<T>) const noexcept {
        Matrix<T, Cols, Rows> pinv{};
        for (std::size_t k = 0; k < K; ++k) {
            if (S[k] <= eps) continue;
            const T inv_s = T{1} / S[k];
            for (std::size_t i = 0; i < Cols; ++i) {
                const T vi = V(i, k) * inv_s;
                for (std::size_t j = 0; j < Rows; ++j) pinv(i, j) += vi * U(j, k);
            }
        }
        return pinv;
    }
};

namespace detail {

template <std::floating_point T, std::size_t M>
constexpr void swap_cols(Matrix<T, M, M>& mat, std::size_t a, std::size_t b) noexcept {
    if (a == b) return;
    for (std::size_t r = 0; r < M; ++r) std::swap(mat(r, a), mat(r, b));
}

template <std::floating_point T, std::size_t R, std::size_t C>
constexpr void swap_cols_rect(Matrix<T, R, C>& mat, std::size_t a, std::size_t b) noexcept {
    if (a == b) return;
    for (std::size_t r = 0; r < R; ++r) std::swap(mat(r, a), mat(r, b));
}

/// Completes columns `[filled, M)` of `U` to an orthonormal basis of R^M
/// using Modified Gram-Schmidt on standard basis vectors.
template <std::floating_point T, std::size_t M>
constexpr void complete_orthonormal_basis(Matrix<T, M, M>& U, std::size_t filled) noexcept {
    for (std::size_t col = filled; col < M; ++col) {
        Vector<T, M> best_vec{};
        T best_norm_sq = T{-1};

        for (std::size_t basis = 0; basis < M; ++basis) {
            Vector<T, M> candidate{};
            candidate[basis] = T{1};

            // Two passes of Modified Gram-Schmidt for numerical stability.
            for (int pass = 0; pass < 2; ++pass) {
                for (std::size_t i = 0; i < col; ++i) {
                    T dot{0};
                    for (std::size_t r = 0; r < M; ++r) dot += candidate[r] * U(r, i);
                    for (std::size_t r = 0; r < M; ++r) candidate[r] -= dot * U(r, i);
                }
            }

            const T norm_sq = candidate.squared_norm();
            if (norm_sq > best_norm_sq) {
                best_norm_sq = norm_sq;
                best_vec = candidate;
            }
        }

        const T inv_norm = T{1} / detail::sqrt(best_norm_sq);
        for (std::size_t r = 0; r < M; ++r) U(r, col) = best_vec[r] * inv_norm;
    }
}

}  // namespace detail

/// Computes the full Singular Value Decomposition of `A`: `A = U * Sigma * V^T`
/// using the one-sided Jacobi algorithm.
template <std::floating_point T, std::size_t Rows, std::size_t Cols>
[[nodiscard]] constexpr SVDResult<T, Rows, Cols> svd(const Matrix<T, Rows, Cols>& A) noexcept {
    if constexpr (Rows < Cols) {
        auto transposed = svd(A.transpose());
        return {transposed.V, transposed.S, transposed.U};
    } else {
        // Tall or square: Rows >= Cols, so K == Cols.
        Matrix<T, Rows, Cols> W = A;
        Matrix<T, Cols, Cols> V = Matrix<T, Cols, Cols>::identity();

        constexpr T eps_tol = std::numeric_limits<T>::epsilon();
        constexpr std::size_t max_sweeps = 30 + 5 * Cols;

        for (std::size_t sweep = 0; sweep < max_sweeps; ++sweep) {
            bool changed = false;
            for (std::size_t p = 0; p + 1 < Cols; ++p) {
                for (std::size_t q = p + 1; q < Cols; ++q) {
                    T a_pp{0}, a_qq{0}, a_pq{0};
                    for (std::size_t r = 0; r < Rows; ++r) {
                        const T wp = W(r, p);
                        const T wq = W(r, q);
                        a_pp += wp * wp;
                        a_qq += wq * wq;
                        a_pq += wp * wq;
                    }

                    if (a_pp == T{0} || a_qq == T{0}) continue;
                    if (detail::abs(a_pq) <= eps_tol * detail::sqrt(a_pp * a_qq)) continue;

                    changed = true;
                    const T tau = (a_qq - a_pp) / (T{2} * a_pq);
                    const T abs_tau = detail::abs(tau);
                    const T t = (tau >= T{0} ? T{1} : T{-1}) /
                                (abs_tau + detail::sqrt(T{1} + tau * tau));
                    const T c = T{1} / detail::sqrt(T{1} + t * t);
                    const T s = t * c;

                    for (std::size_t r = 0; r < Rows; ++r) {
                        const T wp = W(r, p);
                        const T wq = W(r, q);
                        W(r, p) = c * wp - s * wq;
                        W(r, q) = s * wp + c * wq;
                    }
                    for (std::size_t r = 0; r < Cols; ++r) {
                        const T vp = V(r, p);
                        const T vq = V(r, q);
                        V(r, p) = c * vp - s * vq;
                        V(r, q) = s * vp + c * vq;
                    }
                }
            }
            if (!changed) break;
        }

        Vector<T, Cols> S{};
        for (std::size_t j = 0; j < Cols; ++j) {
            T norm_sq{0};
            for (std::size_t r = 0; r < Rows; ++r) norm_sq += W(r, j) * W(r, j);
            S[j] = detail::sqrt(norm_sq);
        }

        // Sort singular values descending and permute columns of W and V accordingly.
        for (std::size_t i = 0; i + 1 < Cols; ++i) {
            std::size_t best = i;
            for (std::size_t j = i + 1; j < Cols; ++j) {
                if (S[j] > S[best]) best = j;
            }
            if (best != i) {
                std::swap(S[i], S[best]);
                detail::swap_cols_rect(W, i, best);
                detail::swap_cols(V, i, best);
            }
        }

        // Normalize non-zero columns of W into U; complete remaining columns if rank-deficient or tall.
        Matrix<T, Rows, Rows> U{};
        const T zero_thresh =
            std::numeric_limits<T>::epsilon() * static_cast<T>(Rows) * (S[0] > T{1} ? S[0] : T{1});
        std::size_t filled = 0;
        for (std::size_t j = 0; j < Cols; ++j) {
            if (S[j] > zero_thresh) {
                const T inv_s = T{1} / S[j];
                for (std::size_t r = 0; r < Rows; ++r) U(r, j) = W(r, j) * inv_s;
                ++filled;
            } else {
                break;
            }
        }
        detail::complete_orthonormal_basis(U, filled);

        return {U, S, V};
    }
}

}  // namespace tensaxon
