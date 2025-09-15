#include "layer.h"
#include "NeuralNetwork.h"
#include "utils.h"

int main(){

    const int epoch = 10;
    const int kernelsize = 3;
    const int classes_num = 2;
    const double learningrate = 0.001;
    double loss;

    NeuralNetwork NNObj;

    Eigen::MatrixXd images;
    NNObj.cv_to_Eigen(cv::imread("images/"), images);

    Eigen::VectorXd PredictedProbability = Eigen::VectorXd::Zero(classes_num);
    Eigen::VectorXd GroundTruth = Eigen::VectorXd::Zero(classes_num);
    GroundTruth(0) = 1;

    Eigen::MatrixXd kernel = Eigen::MatrixXd::Random(kernelsize, kernelsize);
    Eigen::VectorXd conv_output = NNObj.fft_convolution(images, kernel);

    const int inputsize = conv_output.size();
    const int hiddensize = 128;
    const int outputsize = classes_num;

    auto ReLUptr = &Utils::ReLU;
    auto Sigmoidptr = &Utils::Sigmoid;

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

        NNObj.dense(
            olayerptr,
            conv_output,
            PredictedProbability,
            Sigmoidptr);

        loss = NNObj.CrossEntropy(GroundTruth, PredictedProbability);
        std::cout << "Epoch =" << i+1 << "CrossEntropy = " << loss<< std::endl;

        delta_output = NNObj.output_delta(GroundTruth, PredictedProbability);

        NNObj.dense_backward(
            olayerptr,
            PredictedProbability,
            delta_optr,
            dW_optr,
            dB_optr,
            delta_output
        );

        NNObj.dense_backward(
            hlayerptr,
            PredictedProbability,
            nullptr,
            dW_hptr,
            dB_hptr,
            *delta_optr
        );
        
        outputlayer.update_params(dW_optr, dB_optr, learningrate);
        outputlayer.update_params(dW_hptr, dB_hptr, learningrate);

    }
    return 0;
}