#include "LossFunctions.hpp"

namespace loss_functions {

    fp crossentropy(Matrix& predict, Matrix& labels) {
        fp res = static_cast<fp>(0.0);

        size_t m = predict.row(), n = predict.col();

        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < n; j++) {
                res += -labels(i, j) * std::log(predict(i, j));
            }
        }

        return res / static_cast<fp>(n);
    }

}
