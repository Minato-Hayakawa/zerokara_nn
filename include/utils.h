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
            double CrossEntropy(
                Eigen::VectorXd &TargetVector,
                Eigen::VectorXd outVector);
            Eigen::VectorXd output_delta(
                Eigen::VectorXd &y,
                Eigen::VectorXd &t);
            void addition(
                const Eigen::VectorXd &inVector1,
                const Eigen::VectorXd &inVector2,
                Eigen::VectorXd &outVector);
            void multiplication(
                const Eigen::VectorXd &inVector1,
                const Eigen::MatrixXd &inMatrix,
                Eigen::VectorXd &outVector);
    };