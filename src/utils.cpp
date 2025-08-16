#include "utils.h"

class Utils
    {
        public:
    
            std::vector<double> ReLU(std::vector<double> inVector){
                std::vector<double> outVector;
                for (int i=0;i<inVector.size();i++){
                    outVector[i]=std::max(0.0,inVector[i]);
                }return outVector;
            }
            std::vector<double> Sigmoid(std::vector<double> inVector){
                std::vector<double> outVector;
                for (int i=0;i<inVector.size();i++){
                    outVector[i]=0.5 + 0.25 * inVector[i] - 0.010416666666666666 * inVector[i] * inVector[i] * inVector[i];
                }
                return outVector;
            }

            std::vector<double> addition(std::vector<double> inVector1,std::vector<double> inVector2){
                std::vector<double> outVector;
                for (int i=0;i<inVector1.size();i++){
                    outVector[i]=inVector1[i]+inVector2[i];
                }return outVector;
            }
            double *multiplication(double *inp1,double *inp2,const int n,const int m,double *outp){
                for(int i=0;i<n;i++){
                    for(int j=0;j<m;j++){
                        for(int k=0;k<m;k++){
                            outp[i*m +j] += inp1[i*m+k]*inp2[k*m+j];
                        }
                    }
                }return outp;
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