#include "utils.h"

class Utils
    {
        public:
    
            Eigen::VectorXd ReLU(Eigen::VectorXd inVector){
                Eigen::VectorXd outVector(inVector.size());
                for (int i=0;i<inVector.size();i++){
                    outVector[i]=std::max(0.0,inVector[i]);
                }return outVector;
            }
            Eigen::VectorXd Sigmoid(Eigen::VectorXd inVector){
                Eigen::VectorXd outVector(inVector.size());
                for (int i=0;i<inVector.size();i++){
                    outVector[i]= 1.0/(1.0 + std::exp(-inVector[i]));
                }
                return outVector;
            }

            Eigen::VectorXd addition(Eigen::VectorXd inVector1,Eigen::VectorXd inVector2){
                return inVector1+inVector2;
            }
            Eigen::VectorXd multiplication(Eigen::VectorXd inVector1,Eigen::Matrix2Xd inMatrix){
                return inVector1*inMatrix;
            }
            double sum(Eigen::VectorXd inVector){
                double sum = 0;
                for (int i=0;i<inVector.size();i++){
                    sum += inVector[i];
                }return sum;
            }
            double ave(Eigen::VectorXd inVector){
                return sum(inVector)/inVector.size();
            }
            double rms(Eigen::VectorXd inVector){
                Eigen::VectorXd squared_Vector=inVector*inVector;
                return sqrt(ave(squared_Vector));
            }
    };