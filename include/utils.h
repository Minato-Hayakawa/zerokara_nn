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
            double addition(double *inp);
            double multiplication(double *inp);
            double init(double *inp);
    };