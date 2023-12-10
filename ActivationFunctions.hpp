#ifndef ACTIVATIONFUNCTIONS_H
#define ACTIVATIONFUNCTIONS_H
#include <vector>
#include <cmath>
#include <algorithm>
#include "Matrix.hpp"

namespace activation_functions {

    void softmax(Matrix &output) {
        //f(x_i) = e^x_i / sum (e^x_i)

        for(size_t i = 0; i < output.row(); i++) {

            double sum = 0.0, maxEle = -1e9;

            for(size_t j = 0; j < output.col(); j++) {
                maxEle = std::max(maxEle, output(i, j));
            }

            for(size_t j = 0; j < output.col(); j++) {
                output(i, j) = std::exp(output(i, j)-maxEle);
                sum += output(i, j);
            }

            for(size_t j = 0; j < output.col(); j++) {
                output(i, j) /= sum;
            }
        }
    }
    
    void dC_softmax(Matrix &output, Matrix &labels) {

        // dC/dZ = y^_i - y_i
        double num_examples = output.row();
        for(size_t i = 0; i < output.row(); i++) {
            for(size_t j = 0; j < output.col(); j++) {
                if(labels(i, j) == 1)
                    output(i, j) -= 1;

                output(i, j) /= num_examples; //normalization
            }
        }
    }

    void relu(Matrix &output) {

        //f(x) = max(0, x)

        for(size_t i = 0; i < output.row(); i++) {
            for(size_t j = 0; j < output.col(); j++) {
                output(i, j) = std::max(0.0, output(i, j));
            }
        }
    }

    void dC_relu(Matrix &hidden_grad, Matrix &hidden) {

        size_t m = hidden_grad.row(), n = hidden_grad.col();

        for(size_t i = 0; i < m; i++) {
            for(size_t j = 0; j < n; j++) {
                if(hidden(i, j) <= 0.0)
                    hidden_grad(i, j) = 0.0;
            }
        }
    }
}

#endif