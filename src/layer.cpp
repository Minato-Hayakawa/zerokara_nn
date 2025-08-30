#include "layer.h"

layer::layer(int input_size, int output_size)
:gen(rd()), dis(-1.0, 1.0)
{
    weights = Eigen::MatrixXd::NullaryExpr(output_size,
            input_size,
        [&]() {return dis(gen);});
    
    bias = Eigen::VectorXd::NullaryExpr(output_size, [&]() {return dis(gen);});
}

void layer::update_params(
    const Eigen::MatrixXd &dW,
    const Eigen::VectorXd &dB,
    const double learning_rate)
{
    weights+=learning_rate*dW;
    bias+=learning_rate*dB;
}