#include <iostream>
#include <vector>
#include <string>
#include "Layer.hpp"
#include "ActivationFunctions.hpp"

class Dense final: public Layer {
private:
    std::vector<std::vector<double>> weights;
    std::vector<std::vector<double>> bias;
    size_t input_size = 0;
    size_t output_size = 0;
    size_t hidden_size = 0;
    std::string activation = "";

public:
    Dense(size_t input_size, size_t output_size, size_t hidden_size) : 
    input_size(input_size), output_size(output_size), hidden_size(hidden_size) {

        weights.resize(hidden_size, std::vector<double> (output_size, 0.1));
        bias.resize(hidden_size, std::vector<double> (output_size, 0.0));
    }

    Dense(size_t input_size, size_t output_size, size_t hidden_size, std::string activation) :
    input_size(input_size), output_size(output_size), hidden_size(hidden_size), activation(activation) {
        weights.resize(hidden_size, std::vector<double> (output_size, 0.1));
        bias.resize(hidden_size, std::vector<double> (output_size, 0.0));
    }

    std::vector<std::vector<double>> forward(std::vector<std::vector<double>> &input) {
        //std::cout << input.size() << " " << input_size << " " << input[0].size() << " " << hidden_size << std::endl;

        if(input.size() != input_size) {
            throw std::invalid_argument("input sizes do not match");
        }

        if(input[0].size() != hidden_size) {
            throw std::invalid_argument("hidden sizes do not match");
        }

        std::vector<std::vector<double>> output(input_size, std::vector<double> (output_size, 0.0));

        for(size_t i = 0; i < input_size; i++) { 
            for(size_t j = 0; j < output_size; j++) {
                for(size_t k = 0; k < hidden_size; k++) {
                    output[i][j] += input[i][k] * weights[k][j] + bias[k][j];
                }
            }
        }

        if(activation == "softmax")
            ActivationFunctions::softmax(output);

        return output;
    }

    void backward() {
        
    }
};