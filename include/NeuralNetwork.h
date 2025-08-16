#include <string>
#include <vector>
#include <Eigen/Dense>

class NewralNetwork : public Utils{
    private:

        int input_hight,input_width;
        int filter_size;
        int output_hight = input_hight - filter_size + 1;
        int output_width = input_width - filter_size + 1;

    public:

        Eigen::VectorXd dense(Eigen::VectorXd inVector,Eigen::MatrixXd bias,std::string activation);
        double convolution(double *inp,int num);

};