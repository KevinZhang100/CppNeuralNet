#ifndef DENSELAYER_H
#define DENSELAYER_H

#include <string>
#include <vector>
#include <cstdlib>
#include "Layer.hpp"
#include "Matrix.hpp"
#include "Types.hpp"
#include "ActivationFunctions.hpp"
#include "MatrixOperations.hpp"

class Dense final : public Layer {
private:
    Matrix weights;
    Matrix hidden_layer;
    std::vector<fp> bias;

    size_t data_size = 0;
    size_t output_size = 0;
    size_t hidden_size = 0;
    std::string activation;

public:
    Dense(size_t data_size, size_t output_size, size_t hidden_size, std::string activation);

    Matrix forward(Matrix& input) override;
    Matrix backward(Matrix& output_grad, fp learning_rate) override;
};

#endif
