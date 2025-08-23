#include "layer.h"

class layer{
    private:
        std::random_device rd;
        std::mt19937 gen;
        std::uniform_real_distribution<> dis;
    
    public:
        layer(int input_size, int output_size)
        :gen(rd()), dis(-1.0, 1.0)
        {
            Eigen::MatrixXd weights = Eigen::MatrixXd::NullaryExpr(output_size,
                 input_size,
                [&]() {return dis(gen);});
            
            Eigen::VectorXd bias = Eigen::VectorXd::NullaryExpr(output_size, [&]() {return dis(gen);});
        }
};