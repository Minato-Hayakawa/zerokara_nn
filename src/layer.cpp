#include "layer.h"

layer::layer(const int input_size, const int output_size)
:gen(rd())
{
    const double limit = std::sqrt(6/(input_size + output_size));
    std::uniform_real_distribution<> d(-limit, limit);
    weights = Eigen::MatrixXd::NullaryExpr(output_size,
            input_size,
        [&]() {return d(gen);});
    
    bias = Eigen::VectorXd::Zero(output_size);
}

void layer::update_params(
    const Eigen::MatrixXd &dW,
    const Eigen::VectorXd &dB,
    const double learning_rate)
{
    weights-=learning_rate*dW;
    bias-=learning_rate*dB;
}

void layer::update_kernels(
    const Eigen::MatrixXd &dW,
    const double &dB,
    const double learning_rate)
{
    weights-=learning_rate*dW;
    kernel_bias-=learning_rate*dB;
}