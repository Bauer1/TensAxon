#include <cmath>
#include <cstdlib>
#include <iostream>
#include <type_traits>

#include "tensaxon.hpp"

using namespace tensaxon;

// -----------------------------------------------------------------------------
// Minimal test harness. Unlike assert(), CHECK is active in Release builds too.
// -----------------------------------------------------------------------------
namespace {

int g_failures = 0;

template <typename T>
bool approx_equal(T a, T b, T eps = static_cast<T>(1e-5)) {
    return std::abs(a - b) <= eps;
}

}  // namespace

// Variadic so that template arguments containing commas need no extra parens.
#define CHECK(...)                                             \
    do {                                                       \
        if (!(__VA_ARGS__)) {                                  \
            ++g_failures;                                      \
            std::cerr << "  " << __FILE__ << ':' << __LINE__   \
                      << ": CHECK(" #__VA_ARGS__ ") failed\n"; \
        }                                                      \
    } while (0)

#define CHECK_NEAR(a, b, ...) CHECK(approx_equal((a), (b)__VA_OPT__(, ) __VA_ARGS__))

#define TEST_CASE(name) static void name()
#define RUN_TEST(name)                                                                            \
    do {                                                                                          \
        const int before = g_failures;                                                            \
        std::cout << "[ RUN      ] " #name "\n";                                                  \
        name();                                                                                   \
        std::cout << (g_failures == before ? "[       OK ] " : "[  FAILED  ] ") << #name << '\n'; \
    } while (0)

// -----------------------------------------------------------------------------
// Compile-time checks (the whole API is constexpr)
// -----------------------------------------------------------------------------
namespace compile_time {

constexpr Mat3<double> M{{2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0}};
static_assert(M.determinant() == 24.0);
static_assert(Mat3<int>::identity().trace() == 3);
static_assert(Vec3<int>{{1, 0, 0}}.cross(Vec3<int>{{0, 1, 0}}) == Vec3<int>{{0, 0, 1}});
static_assert(Vec2<double>{{3.0, 4.0}}.norm() == 5.0);
static_assert(std::is_same_v<decltype(Matrix<int, 2, 3>{}.transpose()), Matrix<int, 3, 2>>);
static_assert(Tensor<int, 2, 3, 4>::size == 24);
static_assert(Tensor<int, 2, 3, 4>::strides == std::array<std::size_t, 3>{12, 4, 1});
static_assert(
    std::is_same_v<decltype(Tensor<int, 2, 3, 4>{}.permute<2, 0, 1>()), Tensor<int, 4, 2, 3>>);
static_assert(contract<0, 0>(Tensor<int, 3>{{1, 2, 3}}, Tensor<int, 3>{{4, 5, 6}}).value() == 32);

constexpr auto lu_M = lu(M);
static_assert(lu_M.determinant() == 24.0);
static_assert(!lu_M.is_singular());

constexpr auto svd_M = svd(M);
static_assert(svd_M.S[0] == 4.0 && svd_M.S[1] == 3.0 && svd_M.S[2] == 2.0);
static_assert(svd_M.rank() == 3);

}  // namespace compile_time

// -----------------------------------------------------------------------------
// VECTOR
// -----------------------------------------------------------------------------
TEST_CASE(test_vector_arithmetic) {
    Vector<float, 3> v1{{1.0f, 2.0f, 3.0f}};
    Vector<float, 3> v2{{4.0f, 5.0f, 6.0f}};

    auto add = v1 + v2;
    CHECK_NEAR(add[0], 5.0f);
    CHECK_NEAR(add[1], 7.0f);
    CHECK_NEAR(add[2], 9.0f);

    auto sub = v2 - v1;
    CHECK_NEAR(sub[0], 3.0f);
    CHECK_NEAR(sub[1], 3.0f);
    CHECK_NEAR(sub[2], 3.0f);

    auto scaled = v1 * 2.5f;
    CHECK_NEAR(scaled[2], 7.5f);
    CHECK((2.5f * v1) == scaled);

    auto halved = v2 / 2.0f;
    CHECK_NEAR(halved[0], 2.0f);

    auto neg = -v1;
    CHECK_NEAR(neg[1], -2.0f);
}

TEST_CASE(test_vector_dot_norm_cross) {
    Vector<float, 3> v1{{1.0f, 2.0f, -2.0f}};
    Vector<float, 3> v2{{3.0f, 4.0f, 1.0f}};

    CHECK_NEAR(v1.dot(v2), 9.0f);
    CHECK_NEAR(v1.squared_norm(), 9.0f);
    CHECK_NEAR(v1.norm(), 3.0f);
    CHECK_NEAR(v1.normalized().norm(), 1.0f);

    auto c = v1.cross(v2);  // orthogonal to both inputs
    CHECK_NEAR(c.dot(v1), 0.0f);
    CHECK_NEAR(c.dot(v2), 0.0f);
}

// -----------------------------------------------------------------------------
// MATRIX
// -----------------------------------------------------------------------------
TEST_CASE(test_matrix_identity) {
    // Regression: identity() used to loop over Rows*Cols and index out of bounds.
    auto I = Mat4<double>::identity();
    for (std::size_t r = 0; r < 4; ++r)
        for (std::size_t c = 0; c < 4; ++c) CHECK(I(r, c) == (r == c ? 1.0 : 0.0));
}

TEST_CASE(test_matrix_transpose_non_square) {
    // Regression: transpose() used the wrong result type (failed for non-square).
    Matrix<int, 2, 3> A{{1, 2, 3, 4, 5, 6}};
    Matrix<int, 3, 2> At = A.transpose();
    CHECK(At(0, 0) == 1);
    CHECK(At(0, 1) == 4);
    CHECK(At(2, 0) == 3);
    CHECK(At(2, 1) == 6);
    CHECK(At.transpose() == A);
}

TEST_CASE(test_matrix_add_sub) {
    // Regression: operator- used operator-= which did not exist.
    Mat2<int> A{{1, 2, 3, 4}};
    Mat2<int> B{{4, 3, 2, 1}};
    CHECK((A + B) == Mat2<int>::filled(5));
    CHECK((A - B) == (Mat2<int>{{-3, -1, 1, 3}}));
    CHECK((A - A) == Mat2<int>::zeros());
    CHECK((-A) == (Mat2<int>{{-1, -2, -3, -4}}));
    CHECK((A * 2) == (Mat2<int>{{2, 4, 6, 8}}));
}

TEST_CASE(test_matrix_multiplication) {
    Matrix<float, 2, 3> A{{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}};
    Matrix<float, 3, 2> B{{7.0f, 8.0f, 9.0f, 1.0f, 2.0f, 3.0f}};
    auto C = A * B;
    CHECK_NEAR(C(0, 0), 31.0f);
    CHECK_NEAR(C(0, 1), 19.0f);
    CHECK_NEAR(C(1, 0), 85.0f);
    CHECK_NEAR(C(1, 1), 55.0f);
}

TEST_CASE(test_matrix_vector_multiplication) {
    Matrix<float, 2, 3> A{{1.0f, -1.0f, 2.0f, 0.0f, 3.0f, 4.0f}};
    Vector<float, 3> x{{2.0f, 1.0f, -1.0f}};
    auto y = A * x;
    CHECK_NEAR(y[0], -1.0f);
    CHECK_NEAR(y[1], -1.0f);
}

TEST_CASE(test_matrix_row_col_trace) {
    Mat3<int> A{{1, 2, 3, 4, 5, 6, 7, 8, 9}};
    CHECK(A.row(1) == (Vec3<int>{{4, 5, 6}}));
    CHECK(A.col(2) == (Vec3<int>{{3, 6, 9}}));
    CHECK(A.trace() == 15);
}

TEST_CASE(test_matrix_determinants) {
    Mat2<double> M2{{3.0, 8.0, 4.0, 6.0}};
    CHECK_NEAR(M2.determinant(), -14.0);

    Mat3<double> M3{{6.0, 1.0, 1.0, 4.0, -2.0, 5.0, 2.0, 8.0, 7.0}};
    CHECK_NEAR(M3.determinant(), -306.0);

    Mat4<double> M4{
        {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 2.0, 6.0, 4.0, 8.0, 3.0, 1.0, 1.0, 2.0}};
    CHECK_NEAR(M4.determinant(), 72.0);
}

template <typename T, std::size_t N>
static bool is_identity(const Matrix<T, N, N>& m, T eps) {
    for (std::size_t r = 0; r < N; ++r)
        for (std::size_t c = 0; c < N; ++c)
            if (!approx_equal(m(r, c), r == c ? T{1} : T{0}, eps)) return false;
    return true;
}

TEST_CASE(test_matrix_inverse_2x2) {
    // Regression: 2x2 inverse had wrong signs and a wrong diagonal entry.
    Mat2<double> A{{4.0, 7.0, 2.0, 6.0}};
    auto inv = A.inverse();
    CHECK(inv.has_value());
    CHECK_NEAR((*inv)(0, 0), 0.6);
    CHECK_NEAR((*inv)(0, 1), -0.7);
    CHECK_NEAR((*inv)(1, 0), -0.2);
    CHECK_NEAR((*inv)(1, 1), 0.4);
    CHECK(is_identity(A * *inv, 1e-12));
    CHECK(!(Mat2<double>{{1.0, 2.0, 2.0, 4.0}}).inverse().has_value());
}

TEST_CASE(test_matrix_inverse_3x3_and_singular) {
    Mat3<float> A{{1.0f, 2.0f, 3.0f, 0.0f, 1.0f, 4.0f, 5.0f, 6.0f, 0.0f}};
    auto inv = A.inverse();
    CHECK(inv.has_value());
    CHECK(is_identity(A * *inv, 1e-4f));

    Mat3<float> singular{{1.0f, 2.0f, 3.0f, 2.0f, 4.0f, 6.0f, 1.0f, 1.0f, 1.0f}};
    CHECK(!singular.inverse().has_value());
    CHECK_NEAR(singular.determinant(), 0.0f);
}

TEST_CASE(test_matrix_inverse_4x4) {
    Mat4<double> A{
        {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 2.0, 6.0, 4.0, 8.0, 3.0, 1.0, 1.0, 2.0}};
    auto inv = A.inverse();
    CHECK(inv.has_value());
    CHECK(is_identity(A * *inv, 1e-9));
    CHECK(is_identity(*inv * A, 1e-9));
}

// -----------------------------------------------------------------------------
// SOLVER
// -----------------------------------------------------------------------------
TEST_CASE(test_linear_solver) {
    Mat3<double> A{{2.0, 1.0, -1.0, -3.0, -1.0, 2.0, -2.0, 1.0, 2.0}};
    Vec3<double> b{{8.0, -11.0, -3.0}};

    auto x = solve(A, b);
    CHECK(x.has_value());
    CHECK_NEAR((*x)[0], 2.0);
    CHECK_NEAR((*x)[1], 3.0);
    CHECK_NEAR((*x)[2], -1.0);
    CHECK_NEAR((A * *x - b).norm(), 0.0, 1e-6);

    Mat2<double> degA{{1.0, 2.0, 2.0, 4.0}};
    CHECK(!solve(degA, Vec2<double>{{1.0, 2.0}}).has_value());
}

// -----------------------------------------------------------------------------
// TENSOR
// -----------------------------------------------------------------------------
TEST_CASE(test_tensor_indexing) {
    Tensor<int, 2, 3, 4> t{};
    for (std::size_t i = 0; i < t.size; ++i) t.data[i] = static_cast<int>(i);

    CHECK(t.rank == 3);
    CHECK(t(0, 0, 0) == 0);
    CHECK(t(1, 2, 3) == 23);
    CHECK(t(1, 0, 2) == 14);
    CHECK((t[{1, 0, 2}]) == 14);
    CHECK((Tensor<int, 2, 3, 4>::unravel(14) == std::array<std::size_t, 3>{1, 0, 2}));
    CHECK(t.extent<2>() == 4);
}

TEST_CASE(test_tensor_elementwise) {
    auto a = Tensor<double, 2, 2>::filled(3.0);
    auto b = Tensor<double, 2, 2>::ones();
    CHECK((a + b) == Tensor<double, 2, 2>::filled(4.0));
    CHECK((a - b) == Tensor<double, 2, 2>::filled(2.0));
    CHECK((a * 2.0) == Tensor<double, 2, 2>::filled(6.0));
    CHECK((a / 3.0) == b);
    CHECK(hadamard(a, a) == Tensor<double, 2, 2>::filled(9.0));
    CHECK_NEAR(a.sum(), 12.0);
    CHECK_NEAR(a.norm(), 6.0);  // sqrt(4 * 9)
}

TEST_CASE(test_tensor_reshape_permute) {
    Tensor<int, 2, 3> t{{1, 2, 3, 4, 5, 6}};
    auto r = t.reshape<3, 2>();
    CHECK(r(2, 1) == 6);
    CHECK(r(1, 0) == 3);
    CHECK(t.flatten()(4) == 5);

    auto tt = t.transpose();
    CHECK((tt == Tensor<int, 3, 2>{{1, 4, 2, 5, 3, 6}}));

    Tensor<int, 2, 3, 4> u{};
    for (std::size_t i = 0; i < u.size; ++i) u.data[i] = static_cast<int>(i);
    auto p = u.permute<2, 0, 1>();  // shape (4, 2, 3)
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            for (std::size_t k = 0; k < 4; ++k) CHECK(p(k, i, j) == u(i, j, k));
}

TEST_CASE(test_tensor_outer_and_contract) {
    Tensor<int, 2> a{{1, 2}};
    Tensor<int, 3> b{{3, 4, 5}};
    auto o = outer(a, b);  // shape (2, 3)
    static_assert(std::is_same_v<decltype(o), Tensor<int, 2, 3>>);
    CHECK(o(1, 2) == 10);
    CHECK(o(0, 1) == 4);

    // contract<1,0> on rank-2 tensors == matrix product
    Matrix<double, 2, 3> A{{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}};
    Matrix<double, 3, 2> B{{7.0, 8.0, 9.0, 1.0, 2.0, 3.0}};
    auto C = contract<1, 0>(to_tensor(A), to_tensor(B));
    CHECK(to_matrix(C) == A * B);

    // contract<1,0>(matrix, vector) == matrix-vector product
    Vector<double, 3> x{{2.0, 1.0, -1.0}};
    CHECK(to_vector(contract<1, 0>(to_tensor(A), to_tensor(x))) == A * x);

    // contract<0,0>(A, A) == A^T A
    CHECK(to_matrix(contract<0, 0>(to_tensor(A), to_tensor(A))) == A.transpose() * A);

    // Higher-rank: contract the middle axis of a (2,3,4) tensor with a (3) vector.
    Tensor<int, 2, 3, 4> t{};
    for (std::size_t i = 0; i < t.size; ++i) t.data[i] = static_cast<int>(i);
    Tensor<int, 3> v{{1, 10, 100}};
    auto r = contract<1, 0>(t, v);  // shape (2, 4)
    static_assert(std::is_same_v<decltype(r), Tensor<int, 2, 4>>);
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t k = 0; k < 4; ++k)
            CHECK(r(i, k) == t(i, 0, k) * 1 + t(i, 1, k) * 10 + t(i, 2, k) * 100);
}

// -----------------------------------------------------------------------------
// DECOMPOSITIONS (LU & SVD)
// -----------------------------------------------------------------------------
template <typename T, std::size_t R, std::size_t C>
static void check_matrix_near(const Matrix<T, R, C>& a, const Matrix<T, R, C>& b,
                              T eps = static_cast<T>(1e-5)) {
    for (std::size_t r = 0; r < R; ++r)
        for (std::size_t c = 0; c < C; ++c) CHECK_NEAR(a(r, c), b(r, c), eps);
}

TEST_CASE(test_lu_decomposition) {
    // 3x3 requiring pivoting
    Mat3<double> A{{0.0, 2.0, 3.0, 1.0, 4.0, 5.0, 2.0, 1.0, 1.0}};
    auto res = lu(A);

    // P * A == L * U
    check_matrix_near(res.P * A, res.L * res.U);

    // L is unit lower triangular, U is upper triangular
    for (std::size_t r = 0; r < 3; ++r) {
        CHECK_NEAR(res.L(r, r), 1.0);
        for (std::size_t c = r + 1; c < 3; ++c) CHECK_NEAR(res.L(r, c), 0.0);
        for (std::size_t c = 0; c < r; ++c) CHECK_NEAR(res.U(r, c), 0.0);
    }

    CHECK_NEAR(res.determinant(), A.determinant());
    CHECK(!res.is_singular());

    Vec3<double> b{{7.0, 13.0, 4.0}};
    auto x_lu = res.solve(b);
    auto x_direct = solve(A, b);
    CHECK(x_lu.has_value() && x_direct.has_value());
    for (std::size_t i = 0; i < 3; ++i) CHECK_NEAR((*x_lu)[i], (*x_direct)[i]);

    // 4x4 matrix
    Mat4<double> A4{{2.0, 1.0, 1.0, 0.0, 4.0, 3.0, 3.0, 1.0, 8.0, 7.0, 9.0, 5.0, 6.0, 7.0, 9.0, 8.0}};
    auto res4 = lu(A4);
    check_matrix_near(res4.P * A4, res4.L * res4.U);
    CHECK_NEAR(res4.determinant(), A4.determinant());
    Vec4<double> b4{{1.0, 2.0, 3.0, 4.0}};
    auto x4 = res4.solve(b4);
    CHECK(x4.has_value());
    auto Ax4 = A4 * (*x4);
    for (std::size_t i = 0; i < 4; ++i) CHECK_NEAR(Ax4[i], b4[i]);

    // Singular 3x3
    Mat3<double> S{{1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 1.0, 1.0, 1.0}};
    auto res_s = lu(S);
    check_matrix_near(res_s.P * S, res_s.L * res_s.U);
    CHECK(res_s.is_singular());
    CHECK(!res_s.solve(b).has_value());

    // Rectangular 3x2 and 2x3
    Matrix<double, 3, 2> R32{{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}};
    auto res32 = lu(R32);
    check_matrix_near(res32.P * R32, res32.L * res32.U);

    Matrix<double, 2, 3> R23{{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}};
    auto res23 = lu(R23);
    check_matrix_near(res23.P * R23, res23.L * res23.U);
}

TEST_CASE(test_svd_decomposition) {
    // 3x3 general matrix
    Mat3<double> A{{1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 1.0, 0.0, 6.0}};
    auto res = svd(A);
    check_matrix_near(res.U.transpose() * res.U, Mat3<double>::identity());
    check_matrix_near(res.V.transpose() * res.V, Mat3<double>::identity());
    CHECK(res.S[0] >= res.S[1] && res.S[1] >= res.S[2] && res.S[2] >= 0.0);
    check_matrix_near(res.reconstruct(), A);
    CHECK(res.rank() == 3);
    check_matrix_near(A * res.pseudoinverse(), Mat3<double>::identity());

    // Tall 4x2 matrix
    Matrix<double, 4, 2> Tall{{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}};
    auto res_t = svd(Tall);
    check_matrix_near(res_t.U.transpose() * res_t.U, Mat4<double>::identity());
    check_matrix_near(res_t.V.transpose() * res_t.V, Mat2<double>::identity());
    CHECK(res_t.S[0] >= res_t.S[1] && res_t.S[1] > 0.0);
    check_matrix_near(res_t.reconstruct(), Tall);
    check_matrix_near(res_t.pseudoinverse() * Tall, Mat2<double>::identity());

    // Wide 2x4 matrix
    Matrix<double, 2, 4> Wide = Tall.transpose();
    auto res_w = svd(Wide);
    check_matrix_near(res_w.U.transpose() * res_w.U, Mat2<double>::identity());
    check_matrix_near(res_w.V.transpose() * res_w.V, Mat4<double>::identity());
    CHECK(res_w.S[0] >= res_w.S[1] && res_w.S[1] > 0.0);
    check_matrix_near(res_w.reconstruct(), Wide);
    check_matrix_near(Wide * res_w.pseudoinverse(), Mat2<double>::identity());

    // Rank-deficient 3x3 (rank 2)
    Mat3<double> Rank2{{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 7.0, 9.0}};
    auto res_r2 = svd(Rank2);
    check_matrix_near(res_r2.U.transpose() * res_r2.U, Mat3<double>::identity());
    check_matrix_near(res_r2.V.transpose() * res_r2.V, Mat3<double>::identity());
    check_matrix_near(res_r2.reconstruct(), Rank2);
    CHECK(res_r2.rank() == 2);
    auto pinv2 = res_r2.pseudoinverse();
    check_matrix_near(Rank2 * pinv2 * Rank2, Rank2);

    // Zero 3x3 (rank 0)
    Mat3<double> Z = Mat3<double>::zeros();
    auto res_z = svd(Z);
    check_matrix_near(res_z.U.transpose() * res_z.U, Mat3<double>::identity());
    check_matrix_near(res_z.V.transpose() * res_z.V, Mat3<double>::identity());
    check_matrix_near(res_z.reconstruct(), Z);
    CHECK(res_z.rank() == 0);

    // 1x1 negative scalar matrix
    Matrix<double, 1, 1> M1{{-5.0}};
    auto res_1 = svd(M1);
    CHECK_NEAR(res_1.S[0], 5.0);
    check_matrix_near(res_1.U.transpose() * res_1.U, Matrix<double, 1, 1>::identity());
    check_matrix_near(res_1.V.transpose() * res_1.V, Matrix<double, 1, 1>::identity());
    check_matrix_near(res_1.reconstruct(), M1);
}

// -----------------------------------------------------------------------------
// MAIN
// -----------------------------------------------------------------------------
int main() {
    std::cout << "=== Running TensAxon tests ===\n\n";

    RUN_TEST(test_vector_arithmetic);
    RUN_TEST(test_vector_dot_norm_cross);
    RUN_TEST(test_matrix_identity);
    RUN_TEST(test_matrix_transpose_non_square);
    RUN_TEST(test_matrix_add_sub);
    RUN_TEST(test_matrix_multiplication);
    RUN_TEST(test_matrix_vector_multiplication);
    RUN_TEST(test_matrix_row_col_trace);
    RUN_TEST(test_matrix_determinants);
    RUN_TEST(test_matrix_inverse_2x2);
    RUN_TEST(test_matrix_inverse_3x3_and_singular);
    RUN_TEST(test_matrix_inverse_4x4);
    RUN_TEST(test_linear_solver);
    RUN_TEST(test_tensor_indexing);
    RUN_TEST(test_tensor_elementwise);
    RUN_TEST(test_tensor_reshape_permute);
    RUN_TEST(test_tensor_outer_and_contract);
    RUN_TEST(test_lu_decomposition);
    RUN_TEST(test_svd_decomposition);

    if (g_failures != 0) {
        std::cout << '\n' << g_failures << " check(s) FAILED\n";
        return EXIT_FAILURE;
    }
    std::cout << "\nAll tests passed successfully!\n";
    return EXIT_SUCCESS;
}
