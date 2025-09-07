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
    Eigen::VectorXd delta = {};
    Eigen::MatrixXd *dW_hptr = &dW_hidden;
    Eigen::MatrixXd *dW_optr = &dW_output;
    Eigen::VectorXd *dB_hptr = &dB_hidden;
    Eigen::VectorXd *dB_optr = &dB_output;
    Eigen::VectorXd *deltaptr = &delta;

    layer hiddenlayer(inputsize, hiddensize);
    layer outputlayer(hiddensize, outputsize);
    layer *hddnlyrptr = &hiddenlayer;
    layer *outlyrptr = &outputlayer;

    for (int i=0; i<epoch; i++){
        NNObj.dense(
            lyrptr,
            inVector,
            PredictedProbability,
            ReLUptr);
        NNObj.dense(
            lyrptr,
            inVector,
            PredictedProbability,
            ReLUptr);
        NNObj.dense(
            lyrptr,
            inVector,
            PredictedProbability,
            ReLUptr);
        NNObj.dense(
            lyrptr,
            inVector,
            PredictedProbability,
            Sigmoidptr);
        NNObj.dense_backward(
            lyrptr,
            PredictedProbability,
            deltaptr,
            dWptr,
            dBptr,
            NNObj.output_delta(GroundTruth, PredictedProbability)
        );
        lyrObj.update_params(dWptr, dBptr, learnigrate);
        NNObj.dense_backward(
            lyrptr,
            PredictedProbability,
            deltaptr,
            dWptr,
            dBptr,
            NNObj.output_delta(GroundTruth, PredictedProbability)
        );
        lyrObj.update_params(dWptr, dBptr, learnigrate);
        NNObj.dense_backward(
            lyrptr,
            PredictedProbability,
            deltaptr,
            dWptr,
            dBptr,
            NNObj.output_delta(GroundTruth, PredictedProbability)
        );
        lyrObj.update_params(dWptr, dBptr, learnigrate);
        NNObj.dense_backward(
            lyrptr,
            PredictedProbability,
            deltaptr,
            dWptr,
            dBptr,
            NNObj.output_delta(GroundTruth, PredictedProbability)
        );
        lyrObj.update_params(&dW, &dB, learnigrate);
        printf("CrossEntropy = %d\n", NNObj.CrossEntropy(GroundTruth, PredictedProbability));
    }
}