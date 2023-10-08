#include <iostream>
#include <vector>
#include "Model.cpp"

void Testcase1(std::vector<std::vector<double>> &data, std::vector<double> &labels, size_t &m, size_t &n) {
  data = {
    {2, 4, 2, 1},
    {2, 4, 3, 2},
    {3, 3, 2, 3},
    {4, 3, 2, 4}
  };

  labels = {1, 2, 3};
  m = 4, n = 4;
}

void Testcase2(std::vector<std::vector<double>> &data, std::vector<double> &labels, size_t &m, size_t &n) {
    data = {
    {2, 4, 2, 1},
    {2, 4, 3, 2},
    {3, 3, 2},
    {4, 3, 2, 4}
  };

  labels = {1, 2, 3, 4};
  m = 4, n = 4;
}

void Testcase3(std::vector<std::vector<double>> &data, std::vector<double> &labels, size_t &m, size_t &n) {
    data = {
    {2, 4, 2, 1},
    {2, 4, 3, 2},
    {3, 3, 2, 3},
    {4, 3, 2, 4}
  };

  labels = {1, 2, 3, 4};
  m = 4, n = 3;
}

void Testcase4(std::vector<std::vector<double>> &data, std::vector<double> &labels, size_t &m, size_t &n) {
    data = {
    {2, 4, 2, 1},
    {2, 4, 3, 2},
    {3, 3, 2, 3},
    {4, 3, 2, 4}
  };

  labels = {1, 0, 1, 0};
  m = 4, n = 4;
}

int main() {
  std::vector<std::vector<double>> data;
  std::vector<double> labels;
  size_t m = 0, n = 0;

  Testcase4(data, labels, m, n);
  // Testcase2(data, labels, n);
  // Testcase3(data, labels, n);
  // Testcase4(data, labels, n);
  
  //data, labels, input count, input size, epochs
  Model model(data, labels, m, n, 10);
  model.dense(512, "");
  model.dense(256, "");
  model.output();

  std::vector<double> result = model.run();

  for(double num: result) {
    std::cout << num << " ";
  }
  std::cout << std::endl;
}
