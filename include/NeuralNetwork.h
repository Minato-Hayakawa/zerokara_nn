#include <string>

class Perceptron{
        public:

        void dense(std::string activation,bool use_bias,int units);
        void Flatten(double *inp,int num);
};