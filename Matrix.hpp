#ifndef MATRIX_H
#define MATRIX_H
#include <cstddef> 

struct Matrix {
private:
    size_t m = 0, n = 0;
    double **arr = nullptr;

public:
    Matrix () {}

    Matrix(size_t m, size_t n) {
        resize(m, n);
    }

    ~Matrix() {
        del();
    }

    Matrix(const Matrix& other) {
        deep_copy(other);
    }

    double operator()(size_t i,size_t j) const {
        return arr[i][j];
    }

    double& operator()(size_t i, size_t j) {
        return arr[i][j];
    }

    void del() {
        for (size_t i = 0; i < m; i++) {
            delete[] arr[i];
        }

        if(arr)
            delete[] arr;
        
        arr = nullptr;
        m = 0, n = 0;
    }

    void deep_copy(const Matrix& other) {
        del();
        this->m = other.m, this->n = other.n;
        arr = new double*[m];

        for (size_t i = 0; i < m; i++) {
            arr[i] = new double[n];
            
            for (size_t j = 0; j < n; j++) {
                arr[i][j] = other.arr[i][j];
            }
        }
    }

    void resize(size_t m, size_t n) {
        del();

        this->m = m, this->n = n;

        arr = new double*[m];

        for(size_t i = 0; i < m; i++) {

            arr[i] = new double[n];

            for(size_t j = 0; j < n; j++) {
                arr[i][j] = 0.0;
            }
        }
    }

    bool empty() {
        return m == 0 && n == 0;
    }

    size_t row() {
        return m;
    }

    size_t col() {
        return n;
    }
};

#endif
