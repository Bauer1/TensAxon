#include <iostream>
#include <cmath>
#include <cassert>
#include "vector.hpp"
#include "matrix.hpp"

#define TEST_CASE(name) void name()
#define RUN_TEST(name) do { \
    std::cout << "[ RUN      ] " << #name << '\n'; \
    name(); \
    std::cout << "[       OK ] " << #name << '\n'; \
} while(0)

template <typename T>
bool approx_equal(T a, T b, T eps = static_cast<T>(1e-5)) {
    return std::abs(a - b) <= eps;
}

// -----------------------------------------------------------------------------
// VECTOR
// -----------------------------------------------------------------------------

TEST_CASE(test_vector_arithmetic) {
    Vector<float, 3> v1{ {1.0f, 2.0f, 3.0f} };
    Vector<float, 3> v2{ {4.0f, 5.0f, 6.0f} };

    auto add = v1 + v2;
    assert(approx_equal(add[0], 5.0f));
    assert(approx_equal(add[1], 7.0f));
    assert(approx_equal(add[2], 9.0f));

    auto sub = v2 - v1;
    assert(approx_equal(sub[0], 3.0f));
    assert(approx_equal(sub[1], 3.0f));
    assert(approx_equal(sub[2], 3.0f));

    auto scaled = v1 * 2.5f;
    assert(approx_equal(scaled[0], 2.5f));
    assert(approx_equal(scaled[1], 5.0f));
    assert(approx_equal(scaled[2], 7.5f));
}

TEST_CASE(test_vector_dot_and_norm) {
    Vector<float, 3> v1{ {1.0f, 2.0f, -2.0f} };
    Vector<float, 3> v2{ {3.0f, 4.0f, 1.0f} };

    // dot = 1*3 + 2*4 + (-2)*1 = 3 + 8 - 2 = 9
    assert(approx_equal(v1.dot(v2), 9.0f));

    // squared_norm = 1^2 + 2^2 + (-2)^2 = 9
    assert(approx_equal(v1.squared_norm(), 9.0f));
    assert(approx_equal(v1.norm(), 3.0f));
}

// -----------------------------------------------------------------------------
// MATRIX
// -----------------------------------------------------------------------------

TEST_CASE(test_matrix_multiplication) {
    // 2x3 * 3x2 -> 2x2
    Matrix<float, 2, 3> A{ {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    } };

    Matrix<float, 3, 2> B{ {
        7.0f, 8.0f,
        9.0f, 1.0f,
        2.0f, 3.0f
    } };

    auto C = A * B;

    // C(0,0) = 1*7 + 2*9 + 3*2 = 7 + 18 + 6 = 31
    // C(0,1) = 1*8 + 2*1 + 3*3 = 8 + 2 + 9 = 19
    // C(1,0) = 4*7 + 5*9 + 6*2 = 28 + 45 + 12 = 85
    // C(1,1) = 4*8 + 5*1 + 6*3 = 32 + 5 + 18 = 55
    assert(approx_equal(C(0, 0), 31.0f));
    assert(approx_equal(C(0, 1), 19.0f));
    assert(approx_equal(C(1, 0), 85.0f));
    assert(approx_equal(C(1, 1), 55.0f));
}

TEST_CASE(test_matrix_vector_multiplication) {
    Matrix<float, 2, 3> A{ {
        1.0f, -1.0f, 2.0f,
        0.0f,  3.0f, 4.0f
    } };

    Vector<float, 3> x{ {2.0f, 1.0f, -1.0f} };
    auto y = A * x;

    // y0 = 1*2 + (-1)*1 + 2*(-1) = 2 - 1 - 2 = -1
    // y1 = 0*2 + 3*1 + 4*(-1) = 0 + 3 - 4 = -1
    assert(approx_equal(y[0], -1.0f));
    assert(approx_equal(y[1], -1.0f));
}

TEST_CASE(test_matrix_determinants) {
    // 2x2
    Matrix<double, 2, 2> M2{ {
        3.0, 8.0,
        4.0, 6.0
    } };
    assert(approx_equal(M2.determinant(), -14.0)); // 18 - 32

    // 3x3
    Matrix<double, 3, 3> M3{ {
        6.0, 1.0, 1.0,
        4.0, -2.0, 5.0,
        2.0, 8.0, 7.0
    } };
    assert(approx_equal(M3.determinant(), -306.0));

    // 4x4
    Matrix<double, 4, 4> M4{ {
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        2.0, 6.0, 4.0, 8.0,
        3.0, 1.0, 1.0, 2.0
    } };
    assert(approx_equal(M4.determinant(), 72.0));
}

TEST_CASE(test_matrix_inverse_and_singular) {
    Matrix<float, 3, 3> A{ {
        1.0f, 2.0f, 3.0f,
        0.0f, 1.0f, 4.0f,
        5.0f, 6.0f, 0.0f
    } };

    auto invA = A.inverse();
    assert(invA.has_value());

    // A * A^-1 == I
    auto I = A * (*invA);
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            float expected = (r == c) ? 1.0f : 0.0f;
            assert(approx_equal(I(r, c), expected, 1e-4f));
        }
    }

    // 
    Matrix<float, 3, 3> singular{ {
        1.0f, 2.0f, 3.0f,
        2.0f, 4.0f, 6.0f,
        1.0f, 1.0f, 1.0f
    } };
    assert(!singular.inverse().has_value());
    assert(approx_equal(singular.determinant(), 0.0f));
}

// -----------------------------------------------------------------------------
// (solve)
// -----------------------------------------------------------------------------

TEST_CASE(test_linear_solver) {
    // 3x3 
    Matrix<double, 3, 3> A{ {
         2.0,  1.0, -1.0,
        -3.0, -1.0,  2.0,
        -2.0,  1.0,  2.0
    } };
    Vector<double, 3> b{ {8.0, -11.0, -3.0} };

    auto x = solve(A, b);
    assert(x.has_value());

    // Expected solution: x = [2, 3, -1]
    assert(approx_equal((*x)[0], 2.0));
    assert(approx_equal((*x)[1], 3.0));
    assert(approx_equal((*x)[2], -1.0));

    // Check ||A*x - b|| == 0
    auto residual = A * (*x) - b;
    assert(approx_equal(residual.norm(), 0.0, 1e-6));

    // 
    Matrix<double, 2, 2> degA{ {
        1.0, 2.0,
        2.0, 4.0
    } };
    Vector<double, 2> degB{ {1.0, 2.0} };
    assert(!solve(degA, degB).has_value());
}

// -----------------------------------------------------------------------------
// MAIN
// -----------------------------------------------------------------------------

int main() {
    std::cout << "=== Running tests ===\n\n";

    RUN_TEST(test_vector_arithmetic);
    RUN_TEST(test_vector_dot_and_norm);
    RUN_TEST(test_matrix_multiplication);
    RUN_TEST(test_matrix_vector_multiplication);
    RUN_TEST(test_matrix_determinants);
    RUN_TEST(test_matrix_inverse_and_singular);
    RUN_TEST(test_linear_solver);

    std::cout << "\nAll tests passed succesfully!\n";
    return 0;
}