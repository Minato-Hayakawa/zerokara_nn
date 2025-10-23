#include <Eigen/Dense>
#include <random>
class DenseLayer
    {
        private:
            std::random_device rd;
            std::mt19937 gen;
            std::uniform_real_distribution<> dis;

        public:

            Eigen::MatrixXd weights;
            Eigen::VectorXd bias;
            double kernel_bias;
        
            DenseLayer(
                const int input_size,
                const int output_size,
                const int kernel_size);

            void update_params(
                const Eigen::MatrixXd &dW,
                const Eigen::VectorXd &dB,
                const double learning_rate);
    };