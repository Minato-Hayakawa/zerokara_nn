#include <math.h>
#include <numbers>
#include <algorithm>
#include <Eigen/Dense>

class Utils
    {
        public:
            void ReLU(Eigen::VectorXd inVector);
            void Sigmoid(Eigen::VectorXd inVector);

            void addition(Eigen::VectorXd inVector1,Eigen::VectorXd inVector2);
            void multiplication(Eigen::VectorXd inVector1,Eigen::Matrix2Xd inMatrix);

            double sum(Eigen::VectorXd inVector);
            double ave(Eigen::VectorXd inVector);
            double rms(Eigen::VectorXd inVector);
    };