#include "layer.h"
#include "NeuralNetwork.h"
#include "utils.h"
#include <opencv.hpp>
int main(){
    NeuralNetwork NNObj;
    Utils UtlsObj;
    typedef void (Utils::*ReLUptr)(Eigen::MatrixXd inVector, Eigen::MatrixXd outVector);
    typedef void (Utils::*Sigmoidptr)(Eigen::MatrixXd inVector, Eigen::MatrixXd outVector);
    
    Eigen::VectorXd inVector = NNObj.fft_convolution(image, kernel);
    Eigen::VectorXd PredictedProbability;
    Eigen::VectorXd GroundTruth;

    const int epoch = 10;
    const int inputsize = inVector.size();
    const int outputsize = PredictedProbability.size();

    layer lyrObj(inputsize, outputsize);
    for (int i=0; i<epoch; i++){
        NNObj.dense(&lyrObj, inVector, PredictedProbability, void (Utils::ReLU));
        NNObj.dense();
        NNObj.dense();
        NNObj.dense();
        UtlsObj.CrossEntropy(PredictedProbability,GroundTruth)
    }
}