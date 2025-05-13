#ifndef MATRIXOPERATIONS_H
#define MATRIXOPERATIONS_H

#include <iostream>
#include <stdexcept>
#include "Types.hpp"
#include "Matrix.hpp"

namespace operations {

    Matrix multiply(const Matrix& A, const Matrix& B);
    Matrix transpose(const Matrix& A);

}

#endif
