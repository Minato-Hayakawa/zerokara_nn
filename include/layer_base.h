#include <Eigen/Dense>

class LayerBase {
public:

    virtual ~LayerBase() {}
    
    virtual void update_params(double learning_rate) = 0;
    virtual void forward(const Eigen::MatrixXd &input) = 0;
    virtual void backward(const Eigen::MatrixXd &input) = 0;
};