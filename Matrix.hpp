#ifndef MATRIX_H
#define MATRIX_H

#include <cstddef>
#include <cstring>
#include "Types.hpp"

class Matrix {
private:
    size_t m = 0, n = 0;
    fp* arr = nullptr;

public:
    Matrix();
    Matrix(size_t m, size_t n);
    ~Matrix();

    Matrix(const Matrix& other);
    Matrix& operator=(const Matrix& other);

    Matrix(Matrix&& other);
    Matrix& operator=(Matrix&& other);
    void resize(size_t m, size_t n);
    fp* raw();

    inline fp operator()(size_t i, size_t j) const {
        return arr[i * n + j];
    }
    inline fp& operator()(size_t i, size_t j) {
        return arr[i * n + j];
    }
    inline bool empty() const {
        return arr == nullptr || m == 0 || n == 0;
    }
    inline size_t row() const {
        return m;
    }
    inline size_t col() const {
        return n;
    }

private:
    void del();
    void deep_copy(const Matrix& other);
};

#endif
