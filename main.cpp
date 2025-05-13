#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#include "Matrix.hpp"
#include "Model.hpp"
#include "Types.hpp"

void testcase1(Matrix &data, Matrix &labels, size_t &m, size_t &n, size_t &classes) {
    m = 200, n = 2, classes = 2;

    srand(42);
    data.resize(m, n);
    labels.resize(m, classes);

    for (size_t i = 0; i < classes; i++) {
        for (size_t j = i * 100; j < (i + 1) * 100; j++) {
            fp radius = static_cast<fp>(j - i * 100) / static_cast<fp>(100);
            fp rand_component = static_cast<fp>(0.2) * static_cast<fp>((rand() % 100) + 1) / static_cast<fp>(100);
            fp theta = (radius + static_cast<fp>(i)) * static_cast<fp>(4) + rand_component;
            
            data(j, 0) = radius * static_cast<fp>(sin(static_cast<double>(theta)));
            data(j, 1) = radius * static_cast<fp>(cos(static_cast<double>(theta)));

            labels(j, 0) = (i == 0);
            labels(j, 1) = (i == 1);
        }
    }
}

int main() {
    Matrix data;
    Matrix labels;
    size_t m = 0, n = 0, classes = 0;

    testcase1(data, labels, m, n, classes);
    auto start_time = std::chrono::high_resolution_clock::now();

    // data, labels, input count, input size, epochs
    Model model(data, labels, m, n, 5000, classes);
    model.dense(64, "relu");
    model.dense(128, "relu");
    model.dense(64, "relu");
    model.output();

    Matrix result = model.run();

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<fp> time = end_time - start_time;

    // std::cout << "_______________________output________________________" << std::endl;

    // for (size_t i = 0; i < m; i++) {
    //     for (size_t j = 0; j < n; j++) {
    //         std::cout << result(i, j) << " ";
    //     }
    //     std::cout << std::endl;
    // }

    size_t correct = 0;
    for (size_t i = 0; i < m; ++i) {
        size_t pred_class = 0, true_class = 0;
        fp max_pred = result(i, 0), max_true = labels(i, 0);

        for (size_t j = 1; j < classes; ++j) {
            if (result(i, j) > max_pred) {
                max_pred = result(i, j);
                pred_class = j;
            }
            if (labels(i, j) > max_true) {
                max_true = labels(i, j);
                true_class = j;
            }
        }

        if (pred_class == true_class)
            ++correct;
    }

    fp accuracy = static_cast<fp>(correct) / m;
    std::cout << "Accuracy: " << accuracy * 100 << "%" << std::endl;

    std::cout << "Time taken: " << time.count() << " sec" << std::endl;
    return 0;
}
