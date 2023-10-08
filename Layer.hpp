#ifndef LAYER_H
#define LAYER_H
#include <vector> 

class Layer {
public:
    virtual std::vector<std::vector<double>> forward(std::vector<std::vector<double>> &input) = 0;
    virtual void backward() = 0;
};

#endif