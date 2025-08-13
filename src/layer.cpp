#include "layer.h"

std::random_device rd;
std::mt19937 gen(rd);
std::uniform_real_distribution<> dis(-1.0, 1.0);

class layer{
    private:
        int input_size;
        int output_size;
        int width;
        int length;

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
                int width,
                int length,
                std::vector<std::vector<double>> weights){
                    std::vector<double> inner_vector;
                    for (int i=0;i<width;i++){
                        inner_vector.push_back(dis(gen));
                    }
                    weights.push_back(inner_vector);
                }
                
            void bias_init(
                int length,
                std::vector<double> bias){
                    for (int i=0;i<length;i++){
                        bias.push_back(dis(gen));
                    }
                }

};