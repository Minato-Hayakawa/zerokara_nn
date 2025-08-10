class Activation
    {
        private:
            const double e = 2.71828182;
        public:
            double ReLU(double *inp);
            double Sigmoid(double *inp);
            double tanh(double *inp);
    };