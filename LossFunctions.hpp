#ifndef LOSSFUNCTIONS_H
#define LOSSFUNCTIONS_H
#include <vector>
#include <cmath>
#include "Matrix.hpp"

namespace loss_functions {

    double crossentropy(Matrix &predict, Matrix &labels) {

        // - sum(log(x))/n

        double res = 0.0;

        size_t m = predict.row(), n = predict.col();

        for(size_t i = 0; i < m; i++) {
            for(size_t j = 0; j < n; j++) {
                res += -labels(i, j) * std::log(predict(i, j));
            }
        }

        return res/static_cast<double>(n);
    }
}

#endif