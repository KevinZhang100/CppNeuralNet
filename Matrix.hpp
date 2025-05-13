#ifndef MATRIX_H
#define MATRIX_H

#include <cstddef>
#include "Types.hpp"

class Matrix {
private:
    size_t m = 0, n = 0;
    float** arr = nullptr;

public:
    Matrix();
    Matrix(size_t m, size_t n);
    ~Matrix();

    Matrix(const Matrix& other);
    Matrix& operator=(const Matrix& other);

    Matrix(Matrix&& other);
    Matrix& operator=(Matrix&& other);

    float operator()(size_t i, size_t j) const;
    float& operator()(size_t i, size_t j);

    void resize(size_t m, size_t n);
    bool empty() const;
    size_t row() const;
    size_t col() const;

private:
    void del();
    void deep_copy(const Matrix& other);
};

#endif
