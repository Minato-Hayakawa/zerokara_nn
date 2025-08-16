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
            Eigen::VectorXd multiplication(Eigen::VectorXd inVector1,Eigen::Matrix2Xd inMatrix);

            double sum(Eigen::VectorXd inVector);
            double ave(Eigen::VectorXd inVector);
            double rms(Eigen::VectorXd inVector);
    };