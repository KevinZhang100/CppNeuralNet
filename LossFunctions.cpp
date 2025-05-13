#include "LossFunctions.hpp"

namespace loss_functions {

    float crossentropy(Matrix& predict, Matrix& labels) {
        float res = 0.0;

        size_t m = predict.row(), n = predict.col();

        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < n; j++) {
                res += -labels(i, j) * std::log(predict(i, j));
            }
        }

        return res / static_cast<float>(n);
    }

}
