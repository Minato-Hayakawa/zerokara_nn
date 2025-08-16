#include <math.h>
#include <numbers>
// #include <vector>
#include <algorithm>
#include <Eigen/Dense>

class Utils
    {
        public:
            Eigen::VectorXd ReLU(Eigen::VectorXd inVector);
            double Sigmoid(double *inp);
            double tanh(double *inp);

            double addition(double *inp1,double *inp2);
            double *multiplication(double *inp1,double *inp2,const int n,const int m,double *outp);
            double init(double *inp,const int n);
                        double rms(double *inp);

            double sum(double *inp);
            double ave(double *inp);
            double norm(double *inp);
        private:
            double random_uniform();
            double random_gaussian();
    };