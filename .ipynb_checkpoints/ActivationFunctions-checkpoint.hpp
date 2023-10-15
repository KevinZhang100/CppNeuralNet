#ifndef ACTIVATIONFUNCTIONS_H
#define ACTIVATIONFUNCTIONS_H
#include <vector>
#include <cmath>
#include <algorithm>

namespace ActivationFunctions {
    void softmax(std::vector<std::vector<double>> &output) {

        //f(x_i) = e^x_i / sum (e^x_i)

        for(std::vector<double> &vec: output) {

            double sum = 0.0, maxEle = -1e9;

            for(double val: vec) {
                maxEle = std::max(maxEle, val);
            }

            maxEle = maxEle;

            for(double &val: vec) {
                val = std::exp(val-maxEle);
                sum += val;
            }

            for(double &val: vec) {
                val /= sum;
            }
        }
    }
    
    std::vector<std::vector<double>> softmax_crossentropy(std::vector<std::vector<double>> output, std::vector<std::vector<double>> &labels) {

        // dL_i/dF_k = output - (y_i - k)

        for(size_t i = 0; i < output.size(); i++) {
            for(size_t j = 0; j < 2; j++) {
                if(labels[i][j] == 1)
                    output[i][j] -= 1;
            }
        }

        return output;
    }

    void relu(std::vector<std::vector<double>> &output) {

        //f(x) = max(0, x)

        for(std::vector<double> &vec: output) {
            for(double &val: vec) {
                val = std::max(0.1, val);
            }
        }
    }

    void sigmoid(std::vector<std::vector<double>> &output) {

        //f(x) = 1/(1 + e^-x)

        for(std::vector<double> &vec: output) {
            for(double &val: vec) {
                val = 1.0 / 1.0 + exp(-val);
            }
        }
    }

    void sigmoid_derivative(std::vector<std::vector<double>> &output) {

        for(std::vector<double> &vec: output) {
            for(double &val: vec) {
                val = 1 - val;
            }
        }
    }
}

#endif