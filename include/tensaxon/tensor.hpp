#pragma once

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <utility>

#include "detail/math.hpp"
#include "matrix.hpp"
#include "vector.hpp"

namespace tensaxon {

template <Arithmetic T, std::size_t... Dims>
struct Tensor;

namespace detail {

template <std::size_t R>
[[nodiscard]] constexpr std::array<std::size_t, R> row_major_strides(
    const std::array<std::size_t, R>& shape) noexcept {
    std::array<std::size_t, R> strides{};
    std::size_t acc = 1;
    for (std::size_t i = R; i-- > 0;) {
        strides[i] = acc;
        acc *= shape[i];
    }
    return strides;
}

/// Offset into a row-major buffer of the `f`-th element obtained when
/// iterating over every axis except `skip` (row-major order).
template <std::size_t R>
[[nodiscard]] constexpr std::size_t offset_skipping_axis(const std::array<std::size_t, R>& shape,
                                                         const std::array<std::size_t, R>& strides,
                                                         std::size_t skip, std::size_t f) noexcept {
    std::size_t off = 0;
    for (std::size_t i = R; i-- > 0;) {
        if (i == skip) continue;
        off += (f % shape[i]) * strides[i];
        f /= shape[i];
    }
    return off;
}

/// index_sequence of Dims... with axis `Axis` removed.
template <std::size_t Axis, std::size_t... Dims>
struct drop_axis {
    static_assert(Axis < sizeof...(Dims), "Axis out of range");
    static constexpr std::array<std::size_t, sizeof...(Dims)> dims{Dims...};

    template <std::size_t... Is>
    static auto make(std::index_sequence<Is...>)
        -> std::index_sequence<dims[Is < Axis ? Is : Is + 1]...>;

    using type = decltype(make(std::make_index_sequence<sizeof...(Dims) - 1>{}));
};

template <typename T, typename A, typename B>
struct concat_tensor;

template <typename T, std::size_t... A, std::size_t... B>
struct concat_tensor<T, std::index_sequence<A...>, std::index_sequence<B...>> {
    using type = Tensor<T, A..., B...>;
};

template <std::size_t... Ds>
inline constexpr bool all_positive = ((Ds > 0) && ...);

template <std::size_t N, std::size_t... Axes>
[[nodiscard]] consteval bool is_permutation() {
    if (sizeof...(Axes) != N) return false;
    std::array<bool, N> seen{};
    for (std::size_t a : {Axes...}) {
        if (a >= N || seen[a]) return false;
        seen[a] = true;
    }
    return true;
}

}  // namespace detail

/// Fixed-shape, stack-allocated, row-major N-dimensional tensor.
///
/// The shape is part of the type, so shape mismatches are compile errors.
/// `Tensor<T>` (rank 0) is a scalar, `Tensor<T, N>` a vector,
/// `Tensor<T, R, C>` a matrix, and so on.
///
/// @code
///   tensaxon::Tensor<float, 2, 3, 4> t{};   // zero-initialised
///   t(1, 2, 3) = 42.0f;
/// @endcode
template <Arithmetic T, std::size_t... Dims>
struct Tensor {
    static_assert(detail::all_positive<Dims...>, "Tensor dimensions must be positive");

    using value_type = T;
    static constexpr std::size_t rank = sizeof...(Dims);
    static constexpr std::size_t size = (std::size_t{1} * ... * Dims);
    static constexpr std::array<std::size_t, rank> shape{Dims...};
    static constexpr std::array<std::size_t, rank> strides =
        detail::row_major_strides<rank>(std::array<std::size_t, rank>{Dims...});

    std::array<T, size> data{};

    // ----- indexing ---------------------------------------------------------
    /// Row-major flat offset of a multi-index.
    [[nodiscard]] static constexpr std::size_t offset(const std::array<std::size_t, rank>& idx) noexcept {
        std::size_t off = 0;
        for (std::size_t i = 0; i < rank; ++i) {
            assert(idx[i] < shape[i] && "Tensor index out of range");
            off += idx[i] * strides[i];
        }
        return off;
    }

    /// Multi-index of a row-major flat offset.
    [[nodiscard]] static constexpr std::array<std::size_t, rank> unravel(std::size_t flat) noexcept {
        assert(flat < size && "Tensor flat index out of range");
        std::array<std::size_t, rank> idx{};
        for (std::size_t i = rank; i-- > 0;) {
            idx[i] = flat % shape[i];
            flat /= shape[i];
        }
        return idx;
    }

    template <std::integral... Idx>
        requires(sizeof...(Idx) == rank)
    [[nodiscard]] constexpr T& operator()(Idx... idx) noexcept {
        return data[offset({static_cast<std::size_t>(idx)...})];
    }

    template <std::integral... Idx>
        requires(sizeof...(Idx) == rank)
    [[nodiscard]] constexpr const T& operator()(Idx... idx) const noexcept {
        return data[offset({static_cast<std::size_t>(idx)...})];
    }

    [[nodiscard]] constexpr T& operator[](const std::array<std::size_t, rank>& idx) noexcept {
        return data[offset(idx)];
    }
    [[nodiscard]] constexpr const T& operator[](const std::array<std::size_t, rank>& idx) const noexcept {
        return data[offset(idx)];
    }

    /// Extent of axis `Axis`.
    template <std::size_t Axis>
        requires(Axis < rank)
    [[nodiscard]] static constexpr std::size_t extent() noexcept {
        return shape[Axis];
    }

    /// Value of a rank-0 (scalar) tensor.
    [[nodiscard]] constexpr T value() const noexcept
        requires(rank == 0)
    {
        return data[0];
    }

    [[nodiscard]] constexpr auto begin() noexcept { return data.begin(); }
    [[nodiscard]] constexpr auto end() noexcept { return data.end(); }
    [[nodiscard]] constexpr auto begin() const noexcept { return data.begin(); }
    [[nodiscard]] constexpr auto end() const noexcept { return data.end(); }

    // ----- factories --------------------------------------------------------
    [[nodiscard]] static constexpr Tensor zeros() noexcept { return Tensor{}; }

    [[nodiscard]] static constexpr Tensor filled(T value) noexcept {
        Tensor t;
        t.data.fill(value);
        return t;
    }

    [[nodiscard]] static constexpr Tensor ones() noexcept { return filled(T{1}); }

    // ----- compound assignment ----------------------------------------------
    constexpr Tensor& operator+=(const Tensor& other) noexcept {
        for (std::size_t i = 0; i < size; ++i) data[i] += other.data[i];
        return *this;
    }
    constexpr Tensor& operator-=(const Tensor& other) noexcept {
        for (std::size_t i = 0; i < size; ++i) data[i] -= other.data[i];
        return *this;
    }
    constexpr Tensor& operator*=(T scalar) noexcept {
        for (auto& x : data) x *= scalar;
        return *this;
    }
    constexpr Tensor& operator/=(T scalar) noexcept {
        assert(scalar != T{0} && "Division by zero");
        for (auto& x : data) x /= scalar;
        return *this;
    }

    [[nodiscard]] friend constexpr bool operator==(const Tensor&, const Tensor&) = default;

    // ----- reductions -------------------------------------------------------
    [[nodiscard]] constexpr T sum() const noexcept {
        T s{0};
        for (const auto& x : data) s += x;
        return s;
    }

    [[nodiscard]] constexpr T squared_norm() const noexcept {
        T s{0};
        for (const auto& x : data) s += x * x;
        return s;
    }

    /// Frobenius norm.
    [[nodiscard]] constexpr T norm() const noexcept
        requires std::floating_point<T>
    {
        return detail::sqrt(squared_norm());
    }

    // ----- shape manipulation -----------------------------------------------
    /// Same data, new shape. Total size must be unchanged.
    template <std::size_t... NewDims>
    [[nodiscard]] constexpr Tensor<T, NewDims...> reshape() const noexcept {
        static_assert((std::size_t{1} * ... * NewDims) == size,
                      "reshape() must preserve the total number of elements");
        Tensor<T, NewDims...> r{};
        r.data = data;
        return r;
    }

    /// Flattens to a rank-1 tensor.
    [[nodiscard]] constexpr Tensor<T, size> flatten() const noexcept { return reshape<size>(); }

    /// Reorders axes: result axis k is source axis Axes[k].
    /// e.g. for a rank-2 tensor, permute<1, 0>() is the transpose.
    template <std::size_t... Axes>
    [[nodiscard]] constexpr auto permute() const noexcept {
        static_assert(detail::is_permutation<rank, Axes...>(),
                      "permute<Axes...>() requires a permutation of 0..rank-1");
        using Result = Tensor<T, shape[Axes]...>;
        constexpr std::array<std::size_t, rank> axes{Axes...};

        Result r{};
        for (std::size_t flat = 0; flat < size; ++flat) {
            const auto out_idx = Result::unravel(flat);
            std::array<std::size_t, rank> src_idx{};
            for (std::size_t k = 0; k < rank; ++k) src_idx[axes[k]] = out_idx[k];
            r.data[flat] = data[offset(src_idx)];
        }
        return r;
    }

    /// Transpose of a rank-2 tensor.
    [[nodiscard]] constexpr auto transpose() const noexcept
        requires(rank == 2)
    {
        return permute<1, 0>();
    }
};

// ----- element-wise free operators -------------------------------------------
template <typename T, std::size_t... D>
[[nodiscard]] constexpr Tensor<T, D...> operator+(Tensor<T, D...> lhs, const Tensor<T, D...>& rhs) noexcept {
    return lhs += rhs;
}

template <typename T, std::size_t... D>
[[nodiscard]] constexpr Tensor<T, D...> operator-(Tensor<T, D...> lhs, const Tensor<T, D...>& rhs) noexcept {
    return lhs -= rhs;
}

template <typename T, std::size_t... D>
[[nodiscard]] constexpr Tensor<T, D...> operator-(Tensor<T, D...> t) noexcept {
    for (auto& x : t.data) x = -x;
    return t;
}

template <typename T, std::size_t... D>
[[nodiscard]] constexpr Tensor<T, D...> operator*(Tensor<T, D...> t, T scalar) noexcept {
    return t *= scalar;
}

template <typename T, std::size_t... D>
[[nodiscard]] constexpr Tensor<T, D...> operator*(T scalar, Tensor<T, D...> t) noexcept {
    return t *= scalar;
}

template <typename T, std::size_t... D>
[[nodiscard]] constexpr Tensor<T, D...> operator/(Tensor<T, D...> t, T scalar) noexcept {
    return t /= scalar;
}

/// Element-wise (Hadamard) product.
template <typename T, std::size_t... D>
[[nodiscard]] constexpr Tensor<T, D...> hadamard(Tensor<T, D...> lhs, const Tensor<T, D...>& rhs) noexcept {
    for (std::size_t i = 0; i < lhs.size; ++i) lhs.data[i] *= rhs.data[i];
    return lhs;
}

// ----- tensor products --------------------------------------------------------
/// Outer (tensor) product: shapes are concatenated.
/// outer(Tensor<T, A...>, Tensor<T, B...>) -> Tensor<T, A..., B...>
template <typename T, std::size_t... A, std::size_t... B>
[[nodiscard]] constexpr Tensor<T, A..., B...> outer(const Tensor<T, A...>& a,
                                                   const Tensor<T, B...>& b) noexcept {
    Tensor<T, A..., B...> r{};
    constexpr std::size_t nb = Tensor<T, B...>::size;
    for (std::size_t i = 0; i < a.size; ++i)
        for (std::size_t j = 0; j < nb; ++j) r.data[i * nb + j] = a.data[i] * b.data[j];
    return r;
}

/// Contracts axis `AxisA` of `a` with axis `AxisB` of `b` (sum over the shared
/// index). The result shape is a's shape without AxisA followed by b's shape
/// without AxisB.
///
///   contract<0, 0>(vec, vec)  -> dot product (rank-0 tensor)
///   contract<1, 0>(mat, mat)  -> matrix product
///   contract<1, 0>(mat, vec)  -> matrix-vector product
template <std::size_t AxisA, std::size_t AxisB, typename T, std::size_t... A, std::size_t... B>
[[nodiscard]] constexpr auto contract(const Tensor<T, A...>& a, const Tensor<T, B...>& b) noexcept {
    using TA = Tensor<T, A...>;
    using TB = Tensor<T, B...>;
    static_assert(AxisA < TA::rank, "contract: AxisA out of range");
    static_assert(AxisB < TB::rank, "contract: AxisB out of range");
    static_assert(TA::shape[AxisA] == TB::shape[AxisB],
                  "contract: contracted axes must have the same extent");

    using Result = typename detail::concat_tensor<T, typename detail::drop_axis<AxisA, A...>::type,
                                                  typename detail::drop_axis<AxisB, B...>::type>::type;

    constexpr std::size_t K = TA::shape[AxisA];
    constexpr std::size_t free_a = TA::size / K;
    constexpr std::size_t free_b = TB::size / K;
    constexpr std::size_t stride_a = TA::strides[AxisA];
    constexpr std::size_t stride_b = TB::strides[AxisB];

    Result r{};
    for (std::size_t i = 0; i < free_a; ++i) {
        const std::size_t base_a = detail::offset_skipping_axis(TA::shape, TA::strides, AxisA, i);
        for (std::size_t j = 0; j < free_b; ++j) {
            const std::size_t base_b = detail::offset_skipping_axis(TB::shape, TB::strides, AxisB, j);
            T sum{0};
            for (std::size_t k = 0; k < K; ++k)
                sum += a.data[base_a + k * stride_a] * b.data[base_b + k * stride_b];
            r.data[i * free_b + j] = sum;
        }
    }
    return r;
}

// ----- interop with Vector / Matrix (identical row-major layout) -------------
template <typename T, std::size_t N>
[[nodiscard]] constexpr Tensor<T, N> to_tensor(const Vector<T, N>& v) noexcept {
    return Tensor<T, N>{v.data};
}

template <typename T, std::size_t R, std::size_t C>
[[nodiscard]] constexpr Tensor<T, R, C> to_tensor(const Matrix<T, R, C>& m) noexcept {
    return Tensor<T, R, C>{m.data};
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr Vector<T, N> to_vector(const Tensor<T, N>& t) noexcept {
    return Vector<T, N>{t.data};
}

template <typename T, std::size_t R, std::size_t C>
[[nodiscard]] constexpr Matrix<T, R, C> to_matrix(const Tensor<T, R, C>& t) noexcept {
    return Matrix<T, R, C>{t.data};
}

}  // namespace tensaxon
