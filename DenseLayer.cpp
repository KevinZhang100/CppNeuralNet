#include <iostream>
#include <vector>
#include <string>
#include "Layer.hpp"
#include "ActivationFunctions.hpp"
#include "MatrixOperations.hpp"
#include "Matrix.hpp"

class Dense final: public Layer {
private:
    Matrix weights;
    Matrix hidden_layer;
    std::vector<double> bias;

    size_t data_size = 0;
    size_t output_size = 0;
    size_t hidden_size = 0;
    std::string activation;

public:
    Dense(size_t data_size, size_t output_size, size_t hidden_size, std::string activation) :
    data_size(data_size), output_size(output_size), hidden_size(hidden_size), activation(activation) {

        srand(42);
        
        weights.resize(hidden_size, output_size);

        for (size_t i = 0; i < hidden_size; ++i) {
            for (size_t j = 0; j < output_size; ++j) {
                weights(i, j) = 0.01 * (static_cast<double>(std::rand()) / RAND_MAX);
            }
        }

        bias.resize(output_size);
    }

    Matrix forward(Matrix &input) {
        // Input x Weights + Bias
        Matrix output = std::move(operations::multipy(input, weights));

        for(size_t i = 0; i < data_size; i++) {
            for(size_t j = 0; j < output_size; j++) {
                output(i, j) += bias[j];
            }
        }

        hidden_layer = std::move(input);

        if(activation == "softmax")
            activation_functions::softmax(output);
        else if(activation == "relu")
            activation_functions::relu(output);
        
        return output;
    }

    Matrix backward(Matrix& output_grad, double learning_rate) {
        
        Matrix hidden_T = std::move(operations::transpose(hidden_layer));
        Matrix weight_T = std::move(operations::transpose(weights));

        Matrix weight_grad = std::move(operations::multipy(hidden_T, output_grad));
        Matrix hidden_grad = std::move(operations::multipy(output_grad, weight_T));

        for(size_t i = 0; i < output_size; i++) {
            int bias_grad = 0;

            for(size_t j = 0; j < data_size; j++) {
                bias_grad += output_grad(j, i);
            }
            
            bias[i] += learning_rate * -bias_grad;
        }

        for(size_t i = 0; i < hidden_size; i++) {
            for(size_t j = 0; j < output_size; j++) {
                weights(i, j) += learning_rate * -weight_grad(i, j);
            }
        }
        
        activation_functions::dC_relu(hidden_grad, hidden_layer);

        return hidden_grad;
    }
};