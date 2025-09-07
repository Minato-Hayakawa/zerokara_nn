#include "layer.h"
#include "NeuralNetwork.h"
#include "utils.h"
#include "opencv.hpp"

int main(){
    NeuralNetwork NNObj;
    typedef void (Utils::*ReLU)(Eigen::VectorXd &inVector, Eigen::VectorXd &outVector);
    typedef void (Utils::*Sigmoid)(Eigen::VectorXd &inVector, Eigen::VectorXd &outVector);
    ReLU ReLUptr = &Utils::ReLU;
    Sigmoid Sigmoidptr = &Utils::Sigmoid;
    
    Eigen::VectorXd inVector = NNObj.fft_convolution(image, kernel);
    Eigen::VectorXd PredictedProbability;
    Eigen::VectorXd GroundTruth;

    const int epoch = 10;
    const int inputsize = inVector.size();
    const int hiddensize = 128;
    const int outputsize = PredictedProbability.size();
    const double learnigrate = 0.001;

    Eigen::MatrixXd dW_hidden, dW_output;
    Eigen::VectorXd dB_hidden, dB_output;
    Eigen::VectorXd delta_hidden, delta_output;
    Eigen::MatrixXd *dW_hptr = &dW_hidden;
    Eigen::MatrixXd *dW_optr = &dW_output;
    Eigen::VectorXd *dB_hptr = &dB_hidden;
    Eigen::VectorXd *dB_optr = &dB_output;
    Eigen::VectorXd *delta_hptr = &delta_hidden;
    Eigen::VectorXd *delta_optr = &delta_output;

    layer hiddenlayer(inputsize, hiddensize);
    layer outputlayer(hiddensize, outputsize);
    layer *hlayerptr = &hiddenlayer;
    layer *olayerptr = &outputlayer;

    for (int i=0; i<epoch; i++){
        NNObj.dense(
            hlayerptr,
            inVector,
            PredictedProbability,
            ReLUptr);
        delta_hidden = NNObj.output_delta(GroundTruth, PredictedProbability);

        NNObj.dense(
            olayerptr,
            inVector,
            PredictedProbability,
            ReLUptr);
            delta_output = NNObj.output_delta(GroundTruth, PredictedProbability);

        NNObj.dense_backward(
            hlayerptr,
            PredictedProbability,
            delta_hptr,
            dW_hptr,
            dB_hptr,
            NNObj.output_delta(GroundTruth, PredictedProbability)
        );
        hiddenlayer.update_params(dW_hptr, dB_hptr, learnigrate);

        NNObj.dense_backward(
            olayerptr,
            PredictedProbability,
            delta_optr,
            dW_optr,
            dB_optr,
            NNObj.output_delta(GroundTruth, PredictedProbability)
        );
        outputlayer.update_params(dW_optr, dB_optr, learnigrate);
        printf("CrossEntropy = %d\n", NNObj.CrossEntropy(GroundTruth, PredictedProbability));
    }
    return 0;
}