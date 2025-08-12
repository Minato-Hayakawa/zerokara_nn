class Activation
    {
        private:
            const double e = 2.71828182;
        public:
            double ReLU(double *inp){
                if (inp<0){
                    return 0;
                }
                else {
                    return *inp;
                }
            }
            double Sigmoid(double *inp){
                return 1/(1-pow(e,*inp));
            }
            double tanh(double *inp){
                return ((pow(e,*inp)-pow(e,*inp))/(pow(e,*inp)+pow(e,*inp)));
            }
    };

class VectorCaluculation
    {
        public:
            double addition(double *inp);
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