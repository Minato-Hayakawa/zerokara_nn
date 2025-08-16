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
        double convolution(double *inp,int num){
            
        }

};