#include "Matrix.hpp"

Matrix::Matrix() {}

Matrix::Matrix(size_t m, size_t n) {
    resize(m, n);
}

Matrix::~Matrix() {
    del();
}

Matrix::Matrix(Matrix&& other) {
    del();
    m = other.m;
    n = other.n;
    arr = other.arr;
    other.m = 0;
    other.n = 0;
    other.arr = nullptr;
}

Matrix& Matrix::operator=(Matrix&& other) {
    del();
    m = other.m;
    n = other.n;
    arr = other.arr;
    other.m = 0;
    other.n = 0;
    other.arr = nullptr;
    return *this;
}

Matrix::Matrix(const Matrix& other) {
    deep_copy(other);
}

Matrix& Matrix::operator=(const Matrix& other) {
    deep_copy(other);
    return *this;
}

void Matrix::del() {
    delete[] arr;
    arr = nullptr;
    m = 0;
    n = 0;
}

void Matrix::deep_copy(const Matrix& other) {
    resize(other.m, other.n);
    std::memcpy(arr, other.arr, m * n * sizeof(fp));
}

void Matrix::resize(size_t rows, size_t cols) {
    del();
    m = rows;
    n = cols;
    arr = new fp[m * n]();
}

fp* Matrix::raw() {
    return arr;
}