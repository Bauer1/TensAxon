#include <iostream>

#include "tensaxon.hpp"

using namespace tensaxon;

template <typename T, std::size_t R, std::size_t C>
void print(const char* name, const Matrix<T, R, C>& m) {
    std::cout << name << " =\n";
    for (std::size_t r = 0; r < R; ++r) {
        std::cout << "  [";
        for (std::size_t c = 0; c < C; ++c) std::cout << (c ? ", " : "") << m(r, c);
        std::cout << "]\n";
    }
}

int main() {
    // --- Linear system -------------------------------------------------------
    Mat3<double> A{{2.0, 1.0, -1.0, -3.0, -1.0, 2.0, -2.0, 1.0, 2.0}};
    Vec3<double> b{{8.0, -11.0, -3.0}};

    if (auto x = solve(A, b)) {
        std::cout << "solve(A, b) = [" << (*x)[0] << ", " << (*x)[1] << ", " << (*x)[2] << "]\n";
    }
    std::cout << "det(A) = " << A.determinant() << "\n";

    if (auto inv = A.inverse()) print("inverse(A)", *inv);

    // --- Tensors -------------------------------------------------------------
    Tensor<double, 2> u{{1.0, 2.0}};
    Tensor<double, 3> v{{3.0, 4.0, 5.0}};
    auto uv = outer(u, v);  // Tensor<double, 2, 3>
    print("outer(u, v)", to_matrix(uv));

    // Contracting axis 1 of A with axis 0 of b is the matrix-vector product.
    auto Ab = contract<1, 0>(to_tensor(A), to_tensor(b));
    std::cout << "contract<1,0>(A, b) = [" << Ab(0) << ", " << Ab(1) << ", " << Ab(2) << "]\n";

    // Compile-time evaluation.
    constexpr auto I = Mat4<int>::identity();
    static_assert(I.trace() == 4);
    std::cout << "trace(I4) = " << I.trace() << " (computed at compile time)\n";
}
