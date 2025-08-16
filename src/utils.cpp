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
            Eigen::VectorXd multiplication(Eigen::VectorXd inVector1,std::vector<std::vector<double>> inMatrix){

            }
            double init(double *inp,const int n){
                for(int i=0;i<n;i++){
                    inp[i]=0;
                }return *inp;
            }
            double sum(double *inp,int num){
                double sum = 0;
                for (int i=0;i<num;i++){
                    sum += inp[i];
                }return sum;
            }
            double ave(double *inp,int num){
                return sum(inp,num)/num;
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