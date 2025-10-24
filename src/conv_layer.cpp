#include "conv_layer.h"

ConvLayer::ConvLayer(const int kernel_size)
:gen(rd())
{
    const double limit = std::sqrt(6.0 / (kernel_size * kernel_size * 2)); // 簡易版
    std::uniform_real_distribution<> d(-limit, limit);
    kernel_bias = 0.0;

    kernel = Eigen::MatrixXd::NullaryExpr(kernel_size,
                                         kernel_size,
                                         [&]() { return d(gen); });

    kernel_bias = 0.0;
};
void ConvLayer::update_kernels(
    const Eigen::MatrixXd &dW,
    const double dB,
    const double learning_rate)
{
    kernel-=learning_rate*dW;
    kernel_bias-=learning_rate*dB;
};