#include "conv_layer.h"

ConvLayer::ConvLayer(const int kernel_size)
:gen(rd())
{
    const double limit = std::sqrt(6.0 / (kernel_size * kernel_size * 2)); // 簡易版
    std::uniform_real_distribution<> d(-limit, limit);

    this -> kernel = Eigen::MatrixXd::NullaryExpr(kernel_size,
                                         kernel_size,
                                         [&]() { return d(gen); });

    this -> kernel_bias = 0.0;
};
void ConvLayer::update_kernels(
    const double learning_rate)
{
    kernel-=learning_rate*dW;
    kernel_bias-=learning_rate*dB;
};

void ConvLayer::update_params(double learning_rate){
    update_kernels(learning_rate);
}