#include "MatrixOperations.hpp"

namespace operations {
    Matrix multiply(const Matrix& A, const Matrix& B) {
        if (A.col() != B.row()) {
            std::cout << A.col() << " " << B.row() << std::endl;
            throw std::invalid_argument("Matrix A and B sizes do not match");
        }

        size_t a_rows = A.row();
        size_t a_cols = A.col();
        size_t b_cols = B.col();

        Matrix output(a_rows, b_cols);

        for (size_t i = 0; i < a_rows; ++i) {
            for (size_t j = 0; j < b_cols; ++j) {
                fp sum = static_cast<fp>(0);
                for (size_t k = 0; k < a_cols; ++k) {
                    sum += A(i, k) * B(k, j);
                }
                output(i, j) = sum;
            }
        }

        return output;
    }

    Matrix transpose(const Matrix& A) {
        size_t rows = A.row();
        size_t cols = A.col();

        Matrix T(cols, rows);

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                T(j, i) = A(i, j);
            }
        }

        return T;
    }
}
