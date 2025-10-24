#include "dense_layer.h"

class ConvLayer : public DenseLayer{
    public:

        double kernel_bias;
        Eigen::MatrixXd kernel;
        
        std::random_device rd;
        std::mt19937 gen;

        ConvLayer(const int kernel_size);

        void update_kernels(
            const Eigen::MatrixXd &dW,
            const double kernel_bias,
            const double learning_rate);
};