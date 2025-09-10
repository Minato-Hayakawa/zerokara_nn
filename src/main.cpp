#include "layer.h"
#include "NeuralNetwork.h"
#include "utils.h"

int main(){

    const int epoch = 10;
    const int kernelsize = 3;
    const double learnigrate = 0.001;
    double loss;

    NeuralNetwork NNObj;

    Eigen::MatrixXd images;
    NNObj.cv_to_Eigen(cv::imread("images"), images);

    Eigen::VectorXd PredictedProbability;
    Eigen::VectorXd GroundTruth = Eigen::VectorXd::Zero(2);
    GroundTruth(0) = 1;

    Eigen::MatrixXd kernel = Eigen::MatrixXd::Random(kernelsize, kernelsize);
    Eigen::VectorXd conv_output = NNObj.fft_convolution(images, kernel);

    typedef void (Utils::*ReLU)(Eigen::VectorXd &conv_output, Eigen::VectorXd &outVector);
    typedef void (Utils::*Sigmoid)(Eigen::VectorXd &conv_output, Eigen::VectorXd &outVector);
    ReLU ReLUptr = &Utils::ReLU;
    Sigmoid Sigmoidptr = &Utils::Sigmoid;

    const int inputsize = conv_output.size();
    const int hiddensize = 128;
    const int outputsize = PredictedProbability.size();


    layer hiddenlayer(inputsize, hiddensize);
    layer outputlayer(hiddensize, outputsize);
    layer *hlayerptr = &hiddenlayer;
    layer *olayerptr = &outputlayer;

    Eigen::MatrixXd dW_hidden, dW_output;
    Eigen::VectorXd dB_hidden, dB_output;
    Eigen::VectorXd delta_hidden, delta_output;
    Eigen::MatrixXd *dW_hptr = &dW_hidden;
    Eigen::MatrixXd *dW_optr = &dW_output;
    Eigen::VectorXd *dB_hptr = &dB_hidden;
    Eigen::VectorXd *dB_optr = &dB_output;
    Eigen::VectorXd *delta_hptr = &delta_hidden;
    Eigen::VectorXd *delta_optr = &delta_output;

    for (int i=0; i<epoch; i++){
        NNObj.dense(
            hlayerptr,
            conv_output,
            PredictedProbability,
            ReLUptr);
        delta_hidden = NNObj.output_delta(GroundTruth, PredictedProbability);

        NNObj.dense(
            olayerptr,
            conv_output,
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

        loss = NNObj.CrossEntropy(GroundTruth, PredictedProbability);
        std::cout << "Epoch =" << i+1 << "CrossEntropy = " << loss<< std::endl;
    }
    return 0;
}