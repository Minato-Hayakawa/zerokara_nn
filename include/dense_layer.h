#include "layer_base.h"
#include <random>

class DenseLayer : public LayerBase {
public:
    DenseLayer(
        const int input_size,
        const int output_size);

    void forward(
        const Eigen::VectorXd &inVector,
        Eigen::VectorXd &outVector) override;

    Eigen::VectorXd backward(
        const Eigen::VectorXd &delta
    ) override;

    void update_params(double learning_rate) override;

private:

    Eigen::MatrixXd weights;
    Eigen::VectorXd bias;

    Eigen::MatrixXd dW;
    Eigen::VectorXd dB;

    Eigen::VectorXd last_input; 

    std::random_device rd;
    std::mt19937 gen;
};