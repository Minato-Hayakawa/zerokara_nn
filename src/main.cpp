#include "layer.h"
#include "NeuralNetwork.h"
#include "utils.h"
#include <opencv.hpp>
int main(){
    NeuralNetwork NNObj;
    Utils UtlsObj;

    Eigen::VectorXd inVector = NNObj.fft_convolution(image, kernel);
    Eigen::VectorXd outVectror;

    const int epoch = 10;
    const int inputsize = inVector.size();
    const int outputsize = outVector.size();

    layer lyrObj(inputsize, outputsize);
    for (int i=0; i<epoch; i++){
        NNObj.dense();
        NNObj.dense();
        NNObj.dense();
        NNobj.dense();
    }
}