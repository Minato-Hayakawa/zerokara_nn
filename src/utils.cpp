#include "utils.h"

class Utils
    {
        public:
    
            Eigen::VectorXd ReLU(Eigen::VectorXd inVector){
                Eigen::VectorXd outVector;
                for (int i=0;i<inVector.size();i++){
                    outVector[i]=std::max(0.0,inVector[i]);
                }return outVector;
            }
            Eigen::VectorXd Sigmoid(Eigen::VectorXd inVector){
                Eigen::VectorXd outVector;
                for (int i=0;i<inVector.size();i++){
                    outVector[i]=0.5 + 0.25 * inVector[i] - 0.010416666666666666 * inVector[i] * inVector[i] * inVector[i];
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
            double rms(double *inp,int num){
                double squared_inp_sum;
                for (int i=0;i<num;i++){
                    squared_inp_sum+=inp[i]+inp[i];
                }
                return sqrt(squared_inp_sum/num);
            }
            double norm(double *inp,int num){
                double squared_norm;
                for (int i=0;i<num;i++){
                    squared_norm += inp[i]+inp[i];
                }return sqrt(squared_norm);
            }

        private:
            double random_uniform();
            double random_gaussian();
    };