#include "Model.hpp"

Model::Model(Matrix& dat, Matrix& l, size_t m, size_t n, size_t epochs, size_t classes) {
    assert_errors(dat, l, m, n);
    data = dat;
    labels = l;
    this->m = m;
    this->n = n;
    this->epochs = epochs;
    this->classes = classes;
}

void Model::assert_errors(Matrix& data, Matrix& labels, size_t m, size_t n) {
    if (data.empty()) {
        throw std::invalid_argument("data is empty");
    }

    if (labels.empty()) {
        throw std::invalid_argument("labels is empty");
    }

    if (data.row() != labels.row()) {
        std::string error = "data of size " + std::to_string(data.row()) + "\n";
        error += "labels of size " + std::to_string(labels.row()) + "\n";
        error += "sizes do not match\n";

        throw std::invalid_argument(error);
    }
}

void Model::dense(const size_t size, const std::string activation) {
    if (has_out) {
        throw std::invalid_argument("Found dense layer after output layer");
    }
    layer_sizes.push_back(size);
    activations.push_back(activation);
    layerIds.push_back(denseL);
}

void Model::output() {
    if (has_out) {
        throw std::invalid_argument("Cannot add more than one output layer");
    }
    has_out = true;
    layer_sizes.push_back(classes);
    activations.push_back("softmax");
    layerIds.push_back(denseL);
}

Matrix Model::run() {
    if (!has_out) {
        throw std::invalid_argument("Does not have output layer");
    }

    Matrix res;
    size_t prevSize = n;
    std::vector<Layer*> layers;

    std::cout << "Creating Layers." << std::endl;

    for (size_t i = 0; i < layerIds.size(); i++) {
        if (layerIds[i] == denseL) {
            Dense* layer = new Dense(m, layer_sizes[i], prevSize, activations[i]);
            layers.push_back(layer);
        }

        prevSize = layer_sizes[i];
    }

    std::cout << "Running Epochs." << std::endl;

    for (size_t i = 0; i < epochs; i++) {
        Matrix input(data);

        for (Layer* layer : layers) {
            input = layer->forward(input);
        }

        if (i == epochs - 1)
            res = input;

        fp loss = loss_functions::crossentropy(input, labels);
        activation_functions::dC_softmax(input, labels);
        Matrix output_gradient = std::move(input);

        for (int i = layers.size() - 1; i >= 0; i--) {
            output_gradient = layers[i]->backward(output_gradient, learning_rate);
        }

        if (i % 1000 == 0) {
            std::cout << "Epoch " << i + 1 << " finished." << std::endl;
            std::cout << "Loss: " << loss << std::endl;
        }
    }

    for (Layer* layer : layers) {
        delete layer;
    }

    return res;
}

void Model::predict() {
    // Implementation placeholder
}
