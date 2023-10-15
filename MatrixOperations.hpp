#ifndef MATRIXOPERATIONS_H
#define MATRIXOPERATIONS_H
#include <vector>
#include <iostream>
#include "Matrix.hpp"

namespace operations {

    Matrix multipy(Matrix &A, Matrix &B) {
        if(A.col() != B.row())
            throw std::invalid_argument("Matrix A and B sizes do not match");
        
        Matrix output(A.row(), B.col());
        
        for(size_t i = 0; i < A.row(); i++) { 
            for(size_t j = 0; j < B.col(); j++) {
                for(size_t k = 0; k < B.row(); k++) {
                    output(i, j) += A(i, k) * B(k, j);
                }
            }
        }

        return output;
    }


    Matrix transpose(Matrix &A) {

        Matrix T(A.col(), A.row());

        for (size_t i = 0; i < A.row(); i++) {
            for (size_t j = 0; j < A.col(); j++) {
                T(j, i) = A(i, j);
            }
        }
        
        return T;
    }
    
}

#endif
