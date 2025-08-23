#include <Eigen/Dense>
#include <random>
class layer
    {
        public:

            Eigen::MatrixXd weights;
            Eigen::VectorXd bias;
        
            void weights_init(
                int width,
                int length,
                std::vector<std::vector<double>> weights);

            void bias_init(
                const int input_size,
                std::vector<double> bias);;
    };