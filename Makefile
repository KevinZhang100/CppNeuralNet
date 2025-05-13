CXX      = g++
CXXFLAGS = -std=c++23 -Wall -Wextra -fopenmp -O3 
SRCS = main.cpp Model.cpp DenseLayer.cpp Matrix.cpp MatrixOperations.cpp ActivationFunctions.cpp LossFunctions.cpp Layer.cpp

all:
	$(CXX) $(CXXFLAGS) $(SRCS)

clean:
	rm -f a.out
