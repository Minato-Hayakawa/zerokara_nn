#include "layer.h"
#include "NeuralNetwork.h"
#include "utils.h"
#include <opencv.hpp>
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
    const int outputsize = PredictedProbability.size();
    const double learnigrate = 0.001;

    Eigen::MatrixXd dW = {};
    Eigen::VectorXd dB = {};
    Eigen::VectorXd delta = {};
    Eigen::MatrixXd *dWptr = &dW;
    Eigen::VectorXd *dBptr = &dB;
    Eigen::VectorXd *deltaptr = &delta;

    layer lyrObj(inputsize, outputsize);

    for (int i=0; i<epoch; i++){
        NNObj.dense(
            &lyrObj,
            inVector,
            PredictedProbability,
            ReLUptr);
        NNObj.dense(
            &lyrObj,
            inVector,
            PredictedProbability,
            ReLUptr);
        NNObj.dense(
            &lyrObj,
            inVector,
            PredictedProbability,
            ReLUptr);
        NNObj.dense(
            &lyrObj,
            inVector,
            PredictedProbability,
            ReLUptr);
        NNObj.dense_backward(
            &lyrObj,
            PredictedProbability,
            deltaptr,
            dWptr,
            dBptr,
            NNObj.output_delta(GroundTruth, PredictedProbability)
        );
        lyrObj.update_params(dWptr, dBptr, learnigrate);
        NNObj.dense_backward(
            &lyrObj,
            PredictedProbability,
            deltaptr,
            dWptr,
            dBptr,
            NNObj.output_delta(GroundTruth, PredictedProbability)
        );
        lyrObj.update_params(dWptr, dBptr, learnigrate);
        NNObj.dense_backward(
            &lyrObj,
            PredictedProbability,
            deltaptr,
            dWptr,
            dBptr,
            NNObj.output_delta(GroundTruth, PredictedProbability)
        );
        lyrObj.update_params(dWptr, dBptr, learnigrate);
        NNObj.dense_backward(
            &lyrObj,
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