#ifndef ACTIVATIONFUNCTIONS_H
#define ACTIVATIONFUNCTIONS_H

#include <cmath>
#include <algorithm>
#include "Matrix.hpp"
#include "Types.hpp"

namespace activation_functions {

    void softmax(Matrix& output);
    void dC_softmax(Matrix& output, Matrix& labels);
    void relu(Matrix& output);
    void dC_relu(Matrix& hidden_grad, Matrix& hidden);

}

#endif
