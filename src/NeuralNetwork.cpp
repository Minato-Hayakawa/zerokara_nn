#include "NeuralNetwork.h"
#include "utils.h"

class NeuralNetwork : public Utils{

        Eigen::VectorXd dense(Eigen::VectorXd inVector,std::string activation,int units){
            Eigen::MatrixXd bias(inVector.size(),units);
            if (activation == "ReLU"){
                return ReLU(multiplication(inVector,bias));
            }else if (activation == "Sigmoid"){
                return Sigmoid(multiplication(inVector,bias));
            }
        }
        Eigen::MatrixXd convolution(
            Eigen::MatrixXd input_image,
            Eigen::MatrixXd kernel
        ){
            int img_rows = input_image.rows();
            int img_cols = input_image.cols();

            int k_rows = kernel.rows();
            int k_cols = kernel.cols();

            int fft_rows = img_rows + k_rows - 1;
            int fft_cols = img_cols + k_cols - 1;

            Eigen::MatrixXd Padded_image[img_rows+1][img_cols+1];

            for (int i=0; i<input_image.rows; i++){
                for (int j=0; j<input_image.cols; j++){
                    if (i == 0){
                        Padded_image[i][j] = 0;
                    }else if (i == img_rows){
                        Padded_image[i][j] = 0;
                    }else if (j == 0){
                        Padded_image[i][j] = 0;
                    }else if (j == img_cols){
                        Padded_image[i][j] = 0;
                    }
                }
            }

        }

};