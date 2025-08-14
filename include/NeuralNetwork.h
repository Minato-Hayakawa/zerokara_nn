#include <string>

class Perceptron{
    private:

        int input_hight,input_width;
        int filter_size;
        int output_hight = input_hight - filter_size + 1;
        int output_width = input_width - filter_size + 1;

    public:

        void dense(std::string activation,bool use_bias,int units);
        void Flatten(double *inp,int num);
        double convolution(double *inp,int num);

};