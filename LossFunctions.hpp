#ifndef LOSSFUNCTIONS_H
#define LOSSFUNCTIONS_H

#include <cmath>
#include "Matrix.hpp"
#include "Types.hpp"

namespace loss_functions {

    fp crossentropy(Matrix& predict, Matrix& labels);

}

#endif
