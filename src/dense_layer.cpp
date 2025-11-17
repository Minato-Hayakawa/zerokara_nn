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
    const double learning_rate
){
    this -> weights -= learning_rate * this -> dW;
    this -> bias -= learning_rate * this -> dB;
}
void DenseLayer::forward(
    const Eigen::VectorXd &inVector,
    Eigen::VectorXd &outVector
){
        outVector = this -> weights * inVector;
        outVector += this -> bias;
        this -> last_input = inVector;
    }

Eigen::VectorXd DenseLayer::backward(
    const Eigen::VectorXd &delta
){
    this ->  dW = delta * this -> last_input.transpose();
    this -> dB = delta;
    Eigen::VectorXd delta_prev = this -> weights.transpose() * (delta);
    return delta_prev;
}