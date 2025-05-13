#ifndef MODEL_H
#define MODEL_H

#include <string>
#include <vector>
#include <iostream>
#include <cassert>
#include "Matrix.hpp"
#include "Layer.hpp"
#include "DenseLayer.hpp"
#include "LossFunctions.hpp"
#include "ActivationFunctions.hpp"
#include "Types.hpp"

class Model {
private:
    Matrix data;
    Matrix labels;
    std::string loss_function = "binary_crossentropy";

    enum LayerID {
        denseL,
        cnnL
    };

    std::vector<size_t> layer_sizes;
    std::vector<std::string> activations;
    std::vector<LayerID> layerIds;

    size_t n = 0, m = 0, epochs = 0, classes = 0;
    fp learning_rate = 1;
    bool has_out = false;

public:
    Model(Matrix& dat, Matrix& l, size_t m, size_t n, size_t epochs, size_t classes);

    void dense(const size_t size, const std::string activation);
    void output();
    Matrix run();
    void predict();

private:
    void assert_errors(Matrix& data, Matrix& labels, size_t m, size_t n);
};

#endif
