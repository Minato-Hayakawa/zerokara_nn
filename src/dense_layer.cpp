#include "dense_layer.h"

DenseLayer::DenseLayer(const int input_size, const int output_size)
:gen(rd())
{
    const double limit = std::sqrt(6.0/(input_size + output_size));
    std::uniform_real_distribution<> d(-limit, limit);
    weights = Eigen::MatrixXd::NullaryExpr(output_size,
            input_size,
        [&]() {return d(gen);});
    
    bias = Eigen::VectorXd::Zero(output_size);
}

void DenseLayer::update_params(
    const Eigen::MatrixXd &dW,
    const Eigen::VectorXd &dB,
    const double learning_rate)
{
    weights-=learning_rate*dW;
    bias-=learning_rate*dB;
}