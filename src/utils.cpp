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

double CrossEntropy(
    Eigen::VectorXd &TargetVector,
    Eigen::VectorXd outVector){
    double epsilon = 1e-12;
    Eigen::ArrayXd log_out = Eigen::log(outVector.array() + epsilon);
    return -(TargetVector.array()*log_out).sum();
}

Eigen::VectorXd output_delta(
    Eigen::VectorXd &y,
    Eigen::VectorXd &t
){
    return y-t;
}

void Utils::addition(
    Eigen::VectorXd &inVector1,
    Eigen::VectorXd &inVector2,
    Eigen::VectorXd &outVector
){
    outVector = inVector1+inVector2;
}
void Utils::multiplication(
    Eigen::VectorXd &inVector,
    Eigen::MatrixXd &inMatrix,
    Eigen::VectorXd &outVector
){
    outVector = inMatrix*inVector;
}