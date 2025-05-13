#ifndef MATRIXOPERATIONS_H
#define MATRIXOPERATIONS_H

#include <iostream>
#include <stdexcept>
#include "Types.hpp"
#include "Matrix.hpp"

namespace operations {

    Matrix multiply(Matrix& A, Matrix& B);
    Matrix transpose(Matrix& A);

}

#endif
