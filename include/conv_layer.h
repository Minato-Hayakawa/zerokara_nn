#include "layer_base.h"

class ConvLayer : LayerBase{
    public:

        ConvLayer(
            const int input_size,
            const int output_size);

        void forward(
            const Eigen::VectorXd &inVector,
            Eigen::VectorXd &outVector)override;

        void backward(
            const Eigen::VectorXd &delta
        )override;

        void update_params(double learning_rate)override;

        private:
            double kernel_bias;
            Eigen::MatrixXd kernel;
            double dB;
            Eigen::MatrixXd dW;

            std::random_device rd;
            std::mt19937 gen;

            void update_kernels(
                const double learning_rate); 
};