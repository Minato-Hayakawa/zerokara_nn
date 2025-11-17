#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor> 

class LayerBase {
public:
    virtual ~LayerBase() {}

    virtual Eigen::Tensor<double, 4> forward(
        const Eigen::Tensor<double, 4>& input
    ) = 0;

    virtual Eigen::Tensor<double, 4> backward(
        const Eigen::Tensor<double, 4>& delta
    ) = 0;
    
    virtual void update_params(double learning_rate) = 0;
};