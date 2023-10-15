#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include "Model.cpp"
#include "Matrix.hpp"

void testcase1(Matrix &data, Matrix &labels, size_t &m, size_t &n, size_t &classes) {
  m = 200, n = 2, classes = 2;

  srand(42);
  data.resize(m, n);
  labels.resize(m, classes);

  for(size_t i = 0; i < classes; i++) {
    for(size_t j = i*100; j < (i+1)*100; j++) {
      double radius = static_cast<double>((j - i*100)) / 100;
      double theta = (radius+i)*4 + 0.2 * static_cast<double>((rand() % 100) + 1) / 100;;

      data(j, 0) = radius * sin(theta);
      data(j, 1) = radius * cos(theta);

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

  //data, labels, input count, input size, epochs
  Model model(data, labels, m, n, 500, classes);
  model.dense(50, "relu");
  model.dense(25, "relu");
  model.output(); 

  Matrix result(model.run());

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time = end_time - start_time;

  std::cout << "_______________________output________________________" << std::endl;
  
  for(size_t i = 0; i < m; i++) {
    for(size_t j = 0; j < n; j++) {
      std::cout << result(i, j) << " ";
    }
    std::cout << std::endl;
  }

  std::cout << "Time taken: " << time.count() << " s" << std::endl;
}
