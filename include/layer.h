#include<vector>
#include<string>
#include <random>
class layer
    {
        private:
            const int input_size;
            const int output_size;

            std::string activation;
            std::string activation_derivative;
        
        public:
            std::vector<std::vector<double>> weights;
            std::vector<double> bias;

            layer(
                std::vector<std::vector<double>> weights,
                std::vector<double> bias,
                std::string activation,
                std::string activation_derivative);

            void weights_init(
                int width,
                int length,
                std::vector<std::vector<double>> weights);

            void bias_init(
                const int input_size,
                std::vector<double> bias);;
    };