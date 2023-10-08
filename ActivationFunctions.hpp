#ifndef ACTIVATIONFUNCTIONS_H
#define ACTIVATIONFUNCTIONS_H
#include <vector>
#include <cmath>

namespace ActivationFunctions {
    void softmax(std::vector<std::vector<double>> &output) {
        double sum = 0.0, maxEle = -1e9;

        for(std::vector<double> &vec: output) {
            for(double val: vec) {
                maxEle = std::max(maxEle, val);
            }
        }

        maxEle = std::log(maxEle);

        for(std::vector<double> &vec: output) {
            for(double &val: vec) {
                val = std::exp(std::log(val)-maxEle);
                sum += val;
            }
        }

        for(std::vector<double> vec: output) {
            for(double &val: vec) {
                val /= sum;
            }
        }
    }
}

#endif