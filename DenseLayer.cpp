#include "DenseLayer.hpp"

Dense::Dense(size_t data_size, size_t output_size, size_t hidden_size, std::string activation)
    : data_size(data_size), output_size(output_size), hidden_size(hidden_size), activation(activation) {

    srand(42);
    weights.resize(hidden_size, output_size);

    for (size_t i = 0; i < hidden_size; ++i) {
        for (size_t j = 0; j < output_size; ++j) {
            weights(i, j) = static_cast<fp>(0.01) * (static_cast<fp>(std::rand()) / RAND_MAX);
        }
    }

    bias.resize(output_size);
}

Matrix Dense::forward(Matrix& input) {
    Matrix output = operations::multiply(input, weights);

    for (size_t i = 0; i < data_size; i++) {
        for (size_t j = 0; j < output_size; j++) {
            output(i, j) += bias[j];
        }
    }

    hidden_layer = std::move(input);

    if (activation == "softmax")
        activation_functions::softmax(output);
    else if (activation == "relu")
        activation_functions::relu(output);

    return output;
}

Matrix Dense::backward(Matrix& output_grad, fp learning_rate) {
    Matrix hidden_T = operations::transpose(hidden_layer);
    Matrix weight_T = operations::transpose(weights);

    Matrix weight_grad = operations::multiply(hidden_T, output_grad);
    Matrix hidden_grad = operations::multiply(output_grad, weight_T);

    for (size_t i = 0; i < output_size; i++) {
        fp bias_grad = static_cast<fp>(0.0);
        for (size_t j = 0; j < data_size; j++) {
            bias_grad += output_grad(j, i);
        }
        bias[i] += learning_rate * -bias_grad;
    }

    for (size_t i = 0; i < hidden_size; i++) {
        for (size_t j = 0; j < output_size; j++) {
            weights(i, j) += learning_rate * -weight_grad(i, j);
        }
    }

    activation_functions::dC_relu(hidden_grad, hidden_layer);
    return hidden_grad;
}
