#ifndef LAYER_H
#define LAYER_H
#include <vector> 
#include "Matrix.hpp"

class Layer {
public:
    virtual Matrix forward(Matrix &input) = 0;
    virtual Matrix backward(Matrix& output_gradient, double learning_rate) = 0;
    virtual ~Layer() {}
};

#endif