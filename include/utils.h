#include <math.h>
#include <numbers>
#include <algorithm>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "opencv.hpp"

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
                const Eigen::VectorXd &TargetVector,
                const Eigen::VectorXd &outVector);
            Eigen::VectorXd output_delta(
                const Eigen::VectorXd &y,
                const Eigen::VectorXd &t);
            void addition(
                const Eigen::VectorXd &inVector1,
                const Eigen::VectorXd &inVector2,
                Eigen::VectorXd &outVector);
            void multiplication(
                const Eigen::VectorXd &inVector1,
                const Eigen::MatrixXd &inMatrix,
                Eigen::VectorXd &outVector);
            void convert_to_Eigen_tensor(
                const std::vector<cv::Mat> &images,
                Eigen::Tensor<double, 3> &outTensor);
            Eigen::Tensor <double, 3> load_images();
            void convert_tensor_to_matrix(
                const Eigen::Tensor<double, 2> &inTensor,
                Eigen::MatrixXd &outMatrix);
            void convert_matrix_to_tensor(
                const Eigen::MatrixXd &inMatrix,
                Eigen::Tensor<double, 2> &outTensor);
    };