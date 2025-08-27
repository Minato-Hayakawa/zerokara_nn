#include <math.h>
#include <numbers>
#include <algorithm>
#include <Eigen/Dense>

class Utils
    {
        public:
            void ReLU(
                Eigen::VectorXd &inVector,
                Eigen::VectorXd &outVector);
            void Sigmoid(
                Eigen::VectorXd &inVector,
                Eigen::VectorXd &outVector);

            void addition(
                Eigen::VectorXd &inVector1,
                Eigen::VectorXd &inVector2,
                Eigen::VectorXd &outVector);
            void multiplication(
                Eigen::VectorXd &inVector1,
                Eigen::MatrixXd &inMatrix,
                Eigen::VectorXd &outVector);
    };