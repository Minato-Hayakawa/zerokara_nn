#include <Eigen/Dense>
#include <random>

class ConvLayer {
public:
    ConvLayer(const int kernel_size);

    Eigen::MatrixXd forward(const Eigen::MatrixXd& input_image);

    Eigen::MatrixXd backward(const Eigen::MatrixXd& delta_map);

    void update_params(double learning_rate);

private:

    Eigen::MatrixXd kernel;
    double kernel_bias;

    Eigen::MatrixXd dW;
    double dB;

    Eigen::MatrixXd last_input_image;

    std::random_device rd;
    std::mt19937 gen;
};