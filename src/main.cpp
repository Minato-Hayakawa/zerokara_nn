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

    Eigen::Tensor <double, 3> images;
    images = NNObj.load_images();

    Eigen::VectorXd PredictedProbability = Eigen::VectorXd::Zero(classes_num);
    Eigen::VectorXd GroundTruth = Eigen::VectorXd::Zero(classes_num);
    GroundTruth(0) = 1;

    Eigen::Tensor<double, 3> kernel = Eigen::Tensor<double, 3>(kernelsize, kernelsize);
    Eigen::Tensor <double, 3> conv_outputs_Tensor = NNObj.fft_convolution(images, kernel);
    Eigen::VectorXd conv_outputs_Vector;
    const int inputsize = conv_outputs_Tensor.size();
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

    for (int i=0; i<images.size(); i++){
        NNObj.convert_tensor_to_matrix(conv_outputs_Tensor(i), conv_outputs_Vector);
        for (int j=0; i<epoch; j++){
            NNObj.dense(
                hlayerptr,
                conv_outputs_Vector,
                PredictedProbability,
                ReLUptr);

            NNObj.dense(
                olayerptr,
                conv_outputs_Vector,
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


    }
    return 0;
}