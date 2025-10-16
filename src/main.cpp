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
    Eigen::MatrixXd conv_outputs_Matrix;
    Eigen::VectorXd conv_outputs_Vector;
    const int inputsize = conv_outputs_Tensor.size();
    const int hiddensize = 128;
    const int outputsize = classes_num;

    auto ReLUptr = &Utils::ReLU;
    auto Sigmoidptr = &Utils::Sigmoid;

    layer hiddenlayer(inputsize, hiddensize);
    layer outputlayer(hiddensize, outputsize);

    Eigen::MatrixXd dW_hidden, dW_output;
    Eigen::VectorXd dB_hidden, dB_output;
    Eigen::VectorXd delta_hidden, delta_output;
    Eigen::VectorXd hidden_Vector;
    Eigen::VectorXd output_Vector;

    for (int i=0; i<images.dimension(0); i++){
        NNObj.convert_tensor_to_matrix(conv_outputs_Tensor(i), conv_outputs_Matrix);
        NNObj.convert_matrix_to_vector(conv_outputs_Matrix, conv_outputs_Vector);
        for (int j=0; i<epoch; j++){
            NNObj.dense(
                hiddenlayer,
                conv_outputs_Vector,
                hidden_Vector,
                ReLUptr);

            NNObj.dense(
                hiddenlayer,
                hidden_Vector,
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