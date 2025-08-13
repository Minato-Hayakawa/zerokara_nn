#include "layer.h"
#include "utils.h"

class layer{
    private:
        int input_size;
        int output_size;

        std::string activation;
        std::string activation_derivative;
    public:
        std::vector<std::vector<double>> weights;
        std::vector<double> bias;

        layer(
                std::vector<std::vector<double>> weights0,
                std::vector<double> bias0,
                std::string activation0,
                std::string activation_derivative0
            ){
                weights=weights0;
                bias=bias0;
                activation=activation0;
                activation_derivative=activation_derivative0;
                }

            void weights_init(
                const int input_size,
                std::vector<std::vector<double>> weights);
                
            void bias_init(
                const int input_size,
                std::vector<double> bias);

};