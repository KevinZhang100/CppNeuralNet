#include <iostream>
#include <vector>
#include <cassert>
#include "Layer.hpp"
#include "DenseLayer.cpp"

class Model {
private:
    std::vector<std::vector<double>> data;
    std::vector<double> labels;
    
    enum LayerID {
        denseL,
        cnnL,
        lstmL
    };

    std::vector<size_t> layerSizes;
    std::vector<std::string> activations;
    std::vector<LayerID> layerIds;

    size_t n = 0, m = 0, epochs = 0;
    bool hasOut = false;

public:
    Model(std::vector<std::vector<double>> &data, std::vector<double> &labels, size_t m, size_t n, size_t epochs) {
        AssertErrors(data, labels, m, n);
        this->data = data, this->labels = labels, this->m = m, this->n = n,  this->epochs = epochs;
    }

    void AssertErrors(std::vector<std::vector<double>> &data, std::vector<double> &labels, size_t m, size_t n) {
        
        if(data.empty()) {
            throw std::invalid_argument("data is empty");
        }

        if(labels.empty()) {
            throw std::invalid_argument("labels is empty");
        }

         if(data.size() != labels.size()) {
            std::string error = "data of size "  + std::to_string(data.size())+ "\n";
            error += "labels of size "+ std::to_string(labels.size())+ "\n";
            error += "sizes do not match\n";

            throw std::invalid_argument(error);
        }

        for(size_t i; i < data.size(); i++) {
            if(data[i].size() != n) {
                std::string error = "row " + std::to_string(i) + "does not match size " + std::to_string(n);
                throw std::invalid_argument(error);
            }
        }
    }

    void dense(const size_t size, const std::string activation) {
        if(hasOut) {
            throw std::invalid_argument("Found dense layer after output layer");
        }
        layerSizes.push_back(size);
        activations.push_back(activation);
        layerIds.push_back(denseL);
    }

    void output() {
        if(hasOut) {
            throw std::invalid_argument("Cannot add more than one output layer");
        }
        hasOut = true;
        layerSizes.push_back(1);
        activations.push_back("softmax");
        layerIds.push_back(denseL);
    }

    std::vector<double> run() {
        if(!hasOut) {
            throw std::invalid_argument("Does not have output layer");
        }

        std::vector<double> res(n);
        size_t prevSize = n;
        std::vector<Layer*> layers;

        std::cout << "Creating Layers." << std::endl;

        for (size_t i = 0; i < layerIds.size(); i++) {

            if(layerIds[i] == denseL) {
                Dense* layer = new Dense(m, layerSizes[i], prevSize, activations[i]);
                layers.push_back(layer);
            }

            prevSize = layerSizes[i];
        }

        std::cout << "Running Epochs." << std::endl;

        for(size_t i = 0; i < epochs; i++) {

            std::vector<std::vector<double>> dataCopy = data;
            
            for(Layer* layer: layers) {
                dataCopy = layer->forward(dataCopy);
            }

            for(Layer* layer: layers) {
                layer->backward();
            }
            std::cout << "Epoch " << i+1 << " finished." << std::endl;

            for(size_t i = 0; i < n; i++) {
                res[i] = dataCopy[i][0];
            }
        }

        for (Layer* layer : layers) {
            delete layer;
        }

        return res;
    }

    void predict() {

    }
};