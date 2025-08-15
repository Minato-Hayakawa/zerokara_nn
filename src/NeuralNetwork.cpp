#include "NeuralNetwork.h"
#include "utils.h"

class Perceptron : public Utils{

        double dense(double *inp,std::string activation,double *biasp,int units,double *outp){
            if (activation == "ReLU"){
                return ReLU(multiplication(inp,biasp,units,units,outp));
            }else if (activation == "Sigmoid"){
                return Sigmoid(multiplication(inp,biasp,units,units,outp));
            }
        }
        void Flatten(double *inp,int num);
        double convolution(double *inp,int num);

};