#include "layer.h"

layer::layer(const int input_size, const int output_size)
:gen(rd()), dis(-1.0, 1.0)
{
    weights = Eigen::MatrixXd::NullaryExpr(output_size,
            input_size,
        [&]() {return dis(gen);});
    
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
