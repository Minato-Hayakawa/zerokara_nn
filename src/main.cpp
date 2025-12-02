#include "conv_layer.h"
#include "dense_layer.h"
#include "utils.h"

int main(){

    const int epoch = 10;
    const int kernelsize = 3;
    const int classes_num = 2;
    const double learningrate = 0.001;
    double loss;
    ConvLayer convObj(kernelsize);
    DenseLayer denseObj();
    Utils utilsObj;

    Eigen::Tensor <double, 3> images;
    images = utilsObj.load_images();

    Eigen::VectorXd PredictedProbability = Eigen::VectorXd::Zero(classes_num);
    Eigen::VectorXd GroundTruth = Eigen::VectorXd::Zero(classes_num);
    GroundTruth(0) = 1;

    Eigen::Tensor<double, 3> kernel = Eigen::Tensor<double, 3>(images.dimension(0), kernelsize, kernelsize);
    Eigen::Tensor <double, 3> conv_outputs_tensor = convObj.forward(images);

    const int inputsize = conv_outputs_tensor.dimension(1) * conv_outputs_tensor.dimension(1);
    const int hiddensize = 128;
    const int outputsize = classes_num;
    
    DenseLayer denseObj(inputsize, outputsize);
    // Eigen::Tensor <double, 3> conv_outputs_Tensor = NNObj.fft_convolution(images, kernel);
    // Eigen::MatrixXd conv_outputs_Matrix;
    // Eigen::VectorXd conv_outputs_Vector;
    // const int inputsize = conv_outputs_Tensor.dimension(1) * conv_outputs_Tensor.dimension(1);

    auto ReLUptr = &Utils::ReLU;
    auto Sigmoidptr = &Utils::Sigmoid;

    // layer hiddenlayer(inputsize, hiddensize);
    // layer outputlayer(hiddensize, outputsize);

    Eigen::MatrixXd dW_hidden, dW_output;
    Eigen::VectorXd dB_hidden, dB_output;
    Eigen::VectorXd delta_hidden, delta_output;
    Eigen::VectorXd hidden_Vector;
    Eigen::VectorXd output_Vector;

    for (int i=0; i<epoch; i++){
        for (int j=0; j<images.dimension(0); j++){

            convObj.forward(images);

            NNObj.dense(
                hiddenlayer,
                conv_outputs_Vector,
                hidden_Vector,
                ReLUptr);

            NNObj.dense(
                outputlayer,
                hidden_Vector,
                PredictedProbability,
                Sigmoidptr);

            loss = NNObj.CrossEntropy(GroundTruth, PredictedProbability);
            std::cout << "Epoch =" << i+1 << "CrossEntropy = " << loss<< std::endl;

            delta_output = NNObj.output_delta(GroundTruth, PredictedProbability);

            NNObj.dense_backward(
                outputlayer,
                hidden_Vector,
                delta_output,
                dW_output,
                dB_output,
                delta_hidden
            );

            NNObj.dense_backward(
                hiddenlayer,
                conv_outputs_Vector,
                delta_hidden,
                dW_hidden,
                dB_hidden,
                delta_hidden
            );

            outputlayer.update_params(dW_output, dB_output, learningrate);
            hiddenlayer.update_params(dW_hidden, dB_hidden, learningrate);
            }


    }
    return 0;
}