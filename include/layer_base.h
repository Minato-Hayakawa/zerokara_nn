#include <Eigen/Dense>

class LayerBase {
public:

    virtual ~LayerBase() {}
    
    virtual void update_params(double learning_rate) = 0;
    virtual void forward(
        const Eigen::VectorXd &inVector,
        Eigen::VectorXd &outVector) = 0;
    virtual void backward(
        const Eigen::VectorXd &inVector,
        const Eigen::VectorXd &delta,
        Eigen::VectorXd &delta_prev) = 0;
};