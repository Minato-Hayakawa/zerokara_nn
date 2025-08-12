#include <math.h>
class Activation
    {
        private:
            const double e = 2.71828182;
        public:
            double ReLU(double *inp);
            double Sigmoid(double *inp);
            double tanh(double *inp);
    };

class VectorCaluculation
    {
        public:
            double addition(double *inp1,double *inp2);
            double multiplication(double *inp);
            double init(double *inp);
    };

class Random
    {
        private:
            double random_uniform();
            double random_gaussian();
    };

class Auxiliaries
    {
        private:
            double rms(double *inp);
            double sum(double *inp);
            double ave(double *inp);
            double norm(double *inp);
    };