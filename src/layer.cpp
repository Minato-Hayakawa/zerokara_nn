#include "layer.h"

layer::layer(const int input_size, const int output_size)
:gen(rd())
{
    const double limit = std::sqrt(6/(input_size + output_size));
    std::uniform_real_distribution<> d(-limit, limit);
    weights = Eigen::MatrixXd::NullaryExpr(output_size,
            input_size,
        [&]() {return d(gen);});
    
    bias = Eigen::VectorXd::NullaryExpr(output_size, [&]() {return dis(gen);});
}

void layer::update_params(
    Eigen::MatrixXd *dWptr,
    Eigen::VectorXd *dBptr,
    const double learning_rate)
{
    weights-=learning_rate*(*dWptr);
    bias-=learning_rate*(*dBptr);
}
