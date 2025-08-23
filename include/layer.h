#include <Eigen/Dense>
#include <random>
class layer
    {
        public:

            std::vector<std::vector<double>> weights;
            std::vector<double> bias;
        
            void weights_init(
                int width,
                int length,
                std::vector<std::vector<double>> weights);

            void bias_init(
                const int input_size,
                std::vector<double> bias);;
    };