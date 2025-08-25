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
        
            layer(int input_size, int output_size);
            void update_params(
                Eigen::MatrixXd &dW,
                Eigen::VectorXd &dB,
                double learning_rate);
    };