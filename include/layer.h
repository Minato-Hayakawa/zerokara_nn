#include<vector>
class layer
    {
        private:
            const int input_size;
            const int output_size;
        
        public:
            std::vector<std::vector<double>> weights;
            std::vector<double> bias;
    };