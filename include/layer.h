#include <Eigen/Dense>
#include <random>
class layer
    {
        private:
            std::random_device rd;
            std::mt19937 gen;
            std::uniform_real_distribution<> dis;

        public:

            Eigen::MatrixXd weights;
            Eigen::VectorXd bias;
        
            layer(const int input_size, const int output_size);

            void update_params(
                const Eigen::MatrixXd &dW,
                const Eigen::MatrixXd &dB,
                const double learning_rate);

            void update_kernels(
                const Eigen::MatrixXd &dW,
                const double &dB,
                const double learning_rate);
    };