#include "utils.h"


void Utils::ReLU(Eigen::VectorXd &inVector, Eigen::VectorXd &outVector){
    outVector.resize(inVector.size());
    for (int i=0;i<inVector.size();i++){
        outVector[i]=std::max(0.0,inVector[i]);
    }
}
void Utils::Sigmoid(Eigen::VectorXd &inVector, Eigen::VectorXd &outVector){
    outVector.resize(inVector.size());
    for (int i=0;i<inVector.size();i++){
        outVector[i]= 1.0/(1.0 + std::exp(-inVector[i]));
    }
}

void Utils::addition(
    Eigen::VectorXd &inVector1,
    Eigen::VectorXd &inVector2,
    Eigen::VectorXd &outVector
){
    assert(inVector1.size() == inVector2.size());
    outVector.resize(inVector1.size());
    outVector = inVector1+inVector2;
}
void Utils::multiplication(
    Eigen::VectorXd &inVector,
    Eigen::MatrixXd &inMatrix,
    Eigen::VectorXd &outVector
){
    assert(inVector.size() == inMatrix.cols());
    outVector.resize(inVector.size());
    outVector = inMatrix*inVector;
}