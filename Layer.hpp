#ifndef LAYER_H
#define LAYER_H

#include "Matrix.hpp"
#include "Types.hpp"

class Layer {
public:
    virtual Matrix forward(Matrix& input) = 0;
    virtual Matrix backward(Matrix& output_gradient, float learning_rate) = 0;
    virtual ~Layer();
};

#endif
