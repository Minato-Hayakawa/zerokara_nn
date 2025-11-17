#pragma once
#include <Eigen/Dense>
#include <random>

class LayerBase {
public:
    virtual ~LayerBase() {}
    
    virtual void forward(
        const Eigen::VectorXd &inVector,
        Eigen::VectorXd &outVector) = 0;

    virtual Eigen::VectorXd backward(
        const Eigen::VectorXd &delta) = 0;

    virtual void update_params(double learning_rate) = 0;
};