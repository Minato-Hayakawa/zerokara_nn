#include <math.h>
#include <numbers>
// #include <vector>
#include <algorithm>
#include <Eigen/Dense>

class Utils
    {
        public:
            Eigen::VectorXd ReLU(Eigen::VectorXd inVector);
            Eigen::VectorXd Sigmoid(Eigen::VectorXd inVector);

            Eigen::VectorXd addition(Eigen::VectorXd inVector1,Eigen::VectorXd inVector2);
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