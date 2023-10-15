    Matrix(const Matrix& other) {
        resize(other.m, other.n);

        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < n; j++) {
                arr[i][j] = other.arr[i][j];
            }
        }
    }