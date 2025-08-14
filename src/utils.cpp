#include "utils.h"

class Activation
    {
        private:
            const double e = 2.71828182;
        public:
            double ReLU(double *inp){
                if (inp<0){
                    return 0;
                }
                else {
                    return *inp;
                }
            }
            double Sigmoid(double *inp){
                return 1/(1-pow(e,*inp));
            }
            double tanh(double *inp){
                return ((pow(e,*inp)-pow(e,*inp))/(pow(e,*inp)+pow(e,*inp)));
            }
    };

class VectorCaluculation
    {
        public:
            double addition(double *inp1,double *inp2){
                return *inp1+*inp2;
            }
            double multiplication(double *inp1,double *inp2,const int n,const int m,double *outp){
                for(int i=0;i<n;i++){
                    for(int j=0;j<m;j++){
                        for(int k=0;k<m;k++){
                            outp[i*m +j] += inp1[i*m+k]*inp2[k*m+j];
                        }
                    }
                }return *outp;
            }
            double init(double *inp,const int n){
                for(int i=0;i<n;i++){
                    inp[i]=0;
                }return *inp;
            }
    };

class Random
    {
        private:
            double random_uniform();
            double random_gaussian();
    };

class Auxiliaries
    {
        private:
            double sum(double *inp,int num){
                double sum = 0;
                for (int i=0;i<num;i++){
                    sum += inp[i];
                }return sum;
            }
            double ave(double *inp,int num);
            double rms(double *inp);
            double norm(double *inp);
    };