#include "layer.h"

class layer{
    private:
        std::random_device rd;
        std::mt19937 gen;
        std::uniform_real_distribution<> dis;
    
    public:
        Eigen::MatrixXd weights;
        Eigen::VectorXd bias;
        layer(int input_size, int output_size)
        :gen(rd()), dis(-1.0, 1.0)
        {
            Eigen::MatrixXd weights = Eigen::MatrixXd::NullaryExpr(output_size,
                 input_size,
                [&]() {return dis(gen);});
            
            Eigen::VectorXd bias = Eigen::VectorXd::NullaryExpr(output_size, [&]() {return dis(gen);});
        }

        void update_params(
            Eigen::MatrixXd &dW,
            Eigen::VectorXd &dB,
            double learning_rate)
        {
            weights+=learning_rate*dW;
            bias+=learning_rate*dB;
        }
};