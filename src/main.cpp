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
    Utils utilsObj;

    Eigen::Tensor <double, 3> images;
    images = utilsObj.load_images();

    Eigen::VectorXd PredictedProbability = Eigen::VectorXd::Zero(classes_num);
    Eigen::VectorXd GroundTruth = Eigen::VectorXd::Zero(classes_num);
    GroundTruth(0) = 1;

    Eigen::Tensor <double, 3> conv_outputs_tensor;
    Eigen::Tensor <double, 2> input_image;
    Eigen::Tensor<double, 3> kernel = Eigen::Tensor<double, 3>(images.dimension(0), kernelsize, kernelsize);
    const int hiddensize = 128;
    const int outputsize = classes_num;

    const int inputsize = conv_outputs_tensor.dimension(1) * conv_outputs_tensor.dimension(1);
    DenseLayer dense_hiddenObj(inputsize, hiddensize);
    DenseLayer dense_outputObj(hiddensize, outputsize);

    Eigen::MatrixXd dW_hidden, dW_output;
    Eigen::VectorXd dB_hidden, dB_output;
    Eigen::VectorXd delta_hidden, delta_output;
    Eigen::Tensor <double, 3> conv_delta;
    Eigen::Tensor <double, 2> delta_image;
    Eigen::Tensor <double, 2> delta_input;
    Eigen::MatrixXd input_image;
    Eigen::MatrixXd input_matrix;
    Eigen::VectorXd input_vector;
    Eigen::VectorXd hidden_vector;
    Eigen::VectorXd output_vector;


    std::cout<< "Start Training"<< std::endl;
    for (int i=0; i<epoch; i++){
        for (int j=0; j<images.dimension(0); j++){

            conv_outputs_tensor = convObj.forward(images);

            Eigen::Tensor <double, 2> input_image = conv_outputs_tensor(i);
            utilsObj.convert_tensor_to_matrix(input_image, input_matrix);
            utilsObj.convert_matrix_to_vector(input_matrix, input_vector);

            dense_hiddenObj.forward(input_vector, hidden_vector);
            
            dense_outputObj.forward(hidden_vector, output_vector);

            loss = utilsObj.CrossEntropy(GroundTruth, PredictedProbability);
            std::cout << "Epoch =" << i+1 << "CrossEntropy = " << loss<< std::endl;

            delta_output = utilsObj.output_delta(GroundTruth, PredictedProbability);

            delta_hidden =dense_outputObj.backward(delta_output);

            delta_input = dense_outputObj.backward(delta_hidden);

            for (int r=0; r<input_image.dimension(0); r++){
                for (int c=0; c<input_image.dimension(1); c++){
                    conv_delta(j, r, c) = delta_image(r, c);
                }
            }
            dense_hiddenObj.update_params(learningrate);
            dense_outputObj.update_params(learningrate);
            }
            convObj.backward(conv_delta);
            convObj.update_params(learningrate);
    }
    return 0;
}