#include "dense_layer.h"

class ConvLayer : public DenseLayer{
    public:

        double kernel_bias;
        Eigen::MatrixXd kernel;

        void update_kernel(
            const Eigen::MatrixXd &dW,
            const double kernel_bias,
            const double learning_rate);
};