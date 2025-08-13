#include<vector>
class layer
    {
        private:
            const int input_size;
            const int output_size;
            std::vector<double(double)> activation;
            std::vector<double(double)> activation_derivative;
        
        public:
            std::vector<std::vector<double>> weights;
            std::vector<double> bias;
            layer(
                std::vector<std::vector<double>> weights,
                std::vector<double> bias,
                std::vector<double(double)> activation,
                std::vector<double(double)> activation_derivative);
            void weights_init(
                const int input_size,
                std::vector<std::vector<double>> weights);
            void bias_init(
                const int input_size,
                std::vector<double> bias);;
    };