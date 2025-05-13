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

float Matrix::operator()(size_t i, size_t j) const {
    return *(*(arr + i) + j);
}

float& Matrix::operator()(size_t i, size_t j) {
    return *(*(arr + i) + j);
}

void Matrix::del() {
    for (size_t i = 0; i < m; i++) {
        delete[] arr[i];
    }

    if (arr)
        delete[] arr;

    arr = nullptr;
    m = 0;
    n = 0;
}

void Matrix::deep_copy(const Matrix& other) {
    del();
    m = other.m;
    n = other.n;
    arr = new float*[m];

    for (size_t i = 0; i < m; i++) {
        *(arr + i) = new float[n];
        for (size_t j = 0; j < n; j++) {
            *(*(arr + i) + j) = *(*(other.arr + i) + j);
        }
    }
}

void Matrix::resize(size_t m, size_t n) {
    del();

    this->m = m;
    this->n = n;

    arr = new float*[m];
    for (size_t i = 0; i < m; i++) {
        *(arr + i) = new float[n];
        for (size_t j = 0; j < n; j++) {
            *(*(arr + i) + j) = 0.0;
        }
    }
}

bool Matrix::empty() const {
    return m == 0 && n == 0;
}

size_t Matrix::row() const {
    return m;
}

size_t Matrix::col() const {
    return n;
}