#include "utils.h"


void Utils::ReLU(Eigen::VectorXd inVector, Eigen::VectorXd outVector){
    for (int i=0;i<inVector.size();i++){
        outVector[i]=std::max(0.0,inVector[i]);
    }
}
void Utils::Sigmoid(Eigen::VectorXd inVector, Eigen::VectorXd outVector){
    for (int i=0;i<inVector.size();i++){
        outVector[i]= 1.0/(1.0 + std::exp(-inVector[i]));
    }
}

void Utils::addition(
    Eigen::VectorXd inVector1,
    Eigen::VectorXd inVector2,
    Eigen::VectorXd outVector
){
    outVector = inVector1+inVector2;
}
void Utils::multiplication(
    Eigen::VectorXd inVector1,
    Eigen::MatrixXd inMatrix,
    Eigen::MatrixXd outVector
){
    outVector = inVector1*inMatrix;
}

double Utils::sum(Eigen::VectorXd inVector){
    double sum = 0;
    for (int i=0;i<inVector.size();i++){
        sum += inVector[i];
    }return sum;
}
double Utils::ave(Eigen::VectorXd inVector){
    return sum(inVector)/inVector.size();
}
double Utils::rms(Eigen::VectorXd inVector){
    Eigen::VectorXd squared_Vector=inVector.array().square();
    return sqrt(ave(squared_Vector));
}