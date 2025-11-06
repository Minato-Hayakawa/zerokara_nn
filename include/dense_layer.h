#include <Eigen/Dense>
#include <random>
#include "layer_base.h"

class DenseLayer:LayerBase
    {
        private:
            std::random_device rd;
            std::mt19937 gen;
            std::uniform_real_distribution<> dis;
            Eigen::MatrixXd weights;
            Eigen::VectorXd bias;

        public:
        
            DenseLayer(
                const int input_size,
                const int output_size);

            void update_params(double learning_rate)override;
    };