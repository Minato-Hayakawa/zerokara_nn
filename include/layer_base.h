#include <Eigen/Dense>

class LayerBase {
public:

    virtual ~LayerBase() {}
    
    virtual void update_params(double learning_rate) = 0;
};