#include <Eigen/Dense>

class Backpropagation{
    public:
        double ReLU_backward(Eigen::VectorXd inVector,Eigen::VectorXd outVector);
        double Sigmoid_backward(Eigen::VectorXd inVector,Eigen::VectorXd outVector);
};