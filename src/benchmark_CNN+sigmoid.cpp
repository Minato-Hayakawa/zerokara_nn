#include "conv_layer.h"
#include "dense_layer.h"
#include "utils.h"

int main(){

    std::cout << "=== PROGRAM STARTED ===" << std::endl;

    const int epoch = 1;
    const int kernelsize = 3;
    int correct = 0;
    const double learningrate = 100;
    double loss;

    ConvLayer convObj(kernelsize);
    Utils utilsObj;
       
    Eigen::Tensor <double, 3> Training_images, Testing_images;
    Eigen::VectorXd Training_labels, Testing_labels;
    Training_images = utilsObj.load_mnist_images("C:/vscode/C++/zerokara_nn/train-images-idx3-ubyte/train-images.idx3-ubyte");
    Training_labels = utilsObj.load_mnist_labels("C:/vscode/C++/zerokara_nn/train-labels-idx1-ubyte/train-labels.idx1-ubyte");

    Testing_images = utilsObj.load_mnist_images("C:/vscode/C++/zerokara_nn/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte");
    Testing_labels = utilsObj.load_mnist_labels("C:/vscode/C++/zerokara_nn/train-labels-idx1-ubyte/train-labels.idx1-ubyte");


    // std::cout<<"start training" <<std::endl;
    Eigen::Tensor <double, 3> conv_outputs_tensor;
    Eigen::Tensor<double, 3> kernel = Eigen::Tensor<double, 3>(Testing_images.dimension(0), kernelsize, kernelsize);
    const int hiddensize = 128;
    const int outputsize = 10;

    DenseLayer dense_hiddenObj(Testing_images.dimension(1)*Testing_images.dimension(2), hiddensize);
    DenseLayer dense_outputObj(hiddensize, outputsize);

    Eigen::MatrixXd dW_hidden, dW_output;
    Eigen::VectorXd dB_hidden, dB_output;
    Eigen::VectorXd delta_hidden, delta_output;
    Eigen::VectorXd delta_input;
    Eigen::Tensor <double, 3> conv_delta(Testing_images.dimension(0), Testing_images.dimension(1), Testing_images.dimension(2));
    Eigen::MatrixXd input_training_image;
    Eigen::MatrixXd input_testing_image;
    Eigen::VectorXd input_vector;
    Eigen::VectorXd hidden_vector;
    Eigen::VectorXd output_vector;
    int target_idx;
    Eigen::VectorXd ground_truth;


    for (int i=0; i<epoch; i++){

        conv_outputs_tensor = convObj.forward(Training_images);

        for (int j=0; j<Training_images.dimension(0); j++){
            
            target_idx = Training_labels(j);
            ground_truth = Eigen::VectorXd::Zero(10);
            ground_truth(target_idx) = 1.0;
            Eigen::Tensor <double, 2> input_training_images = conv_outputs_tensor.chip(j, 0);
            utilsObj.convert_tensor_to_matrix(input_training_images, input_training_image);
            utilsObj.convert_matrix_to_vector(input_training_image, input_vector);

            dense_hiddenObj.forward(input_vector / 255.0, hidden_vector);
            utilsObj.ReLU(hidden_vector, hidden_vector);
            dense_outputObj.forward(hidden_vector, output_vector);
            utilsObj.Sigmoid(output_vector, output_vector);

            loss = utilsObj.CrossEntropy(output_vector, ground_truth);
            if(j%6000 == 0){
                std::cout << "Epoch =" << i+1 << "\nCrossEntropy = " << loss<< std::endl;
            }
            delta_output = utilsObj.output_delta(ground_truth, output_vector);

            delta_hidden =dense_outputObj.backward(delta_output);

            delta_input = dense_hiddenObj.backward(delta_hidden);

            for (int r=0; r<Training_images.dimension(1); r++){
                for (int c=0; c<Training_images.dimension(2); c++){
                    int vector_idx = r * Training_images.dimension(2) + c;
                    conv_delta(j, r, c) = delta_input(vector_idx);
                }
            }
            dense_hiddenObj.update_params(learningrate);
            dense_outputObj.update_params(learningrate);
            }
            convObj.backward(conv_delta);
            convObj.update_params(learningrate);


        conv_outputs_tensor = convObj.forward(Testing_images);
      
        for (int j=0; j<Testing_images.dimension(0); j++){

            target_idx = Testing_labels(j);
            ground_truth = Eigen::VectorXd::Zero(10);
            ground_truth(target_idx) = 1.0;
            Eigen::Tensor <double, 2> input_testing_images = conv_outputs_tensor.chip(j, 0);
            utilsObj.convert_tensor_to_matrix(input_testing_images, input_testing_image);
            utilsObj.convert_matrix_to_vector(input_testing_image, input_vector);

            dense_hiddenObj.forward(input_vector / 225.0, hidden_vector);
            utilsObj.ReLU(hidden_vector, hidden_vector);
            dense_outputObj.forward(hidden_vector, output_vector);
            utilsObj.Sigmoid(output_vector, output_vector);
            loss = utilsObj.CrossEntropy(ground_truth, output_vector);
            if(j&6000 == 0){
                std::cout << "Epoch =" << i+1 << "CrossEntropy = " << loss<< std::endl;
            }
            delta_output = utilsObj.output_delta(output_vector, ground_truth);

            delta_hidden =dense_outputObj.backward(delta_output);

            delta_input = dense_hiddenObj.backward(delta_hidden);

            for (int r=0; r<Training_images.dimension(1); r++){
                for (int c=0; c<Training_images.dimension(2); c++){
                    int vector_idx = r * Training_images.dimension(2) + c;
                    conv_delta(j, r, c) = delta_input(vector_idx);
                }
            }
            dense_hiddenObj.update_params(learningrate);
            dense_outputObj.update_params(learningrate);

            convObj.backward(conv_delta);
            convObj.update_params(learningrate);
        Eigen::Index maxRow, maxCol;
        output_vector.maxCoeff(&maxRow, &maxCol); 

            if (maxRow == Testing_images(j)) { 
                correct++;
            }
        }
    }
    return 0;

}