#include <math.h>
#include <numbers>
#include <algorithm>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

class Utils
    {
        public:
            int reverseInt(int i);
            int readInt(std::ifstream& file);
            Eigen::Tensor<double, 3> load_mnist_images(std::string path);
            Eigen::VectorXd load_mnist_labels(std::string path);
            void ReLU(
                const Eigen::VectorXd &inVector,
                Eigen::VectorXd &outVector);
            void Sigmoid(
                const Eigen::VectorXd &inVector,
                Eigen::VectorXd &outVector);
            void Softmax(
                const Eigen::VectorXd &inVector,
                Eigen::VectorXd &outVector);
            double CrossEntropy(
                const Eigen::VectorXd &TargetVector,
                const Eigen::VectorXd &outVector);
            Eigen::VectorXd output_delta(
                const Eigen::VectorXd &y,
                const Eigen::VectorXd &t);
            Eigen::VectorXd output_delta_ReLU(
                const Eigen::VectorXd &output,
                const Eigen::VectorXd &ground_truth);
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
            void convert_matrix_to_vector(
                const Eigen::MatrixXd inMatrix,
                Eigen::VectorXd &outVector);
    };