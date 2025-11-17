#include <Eigen/Dense>>
#include <random>
#include <fftw3.h>
#include <unsupported/Eigen/CXX11/Tensor>

class ConvLayer {
public:
    ConvLayer(const int kernel_size);

    Eigen::Tensor<double, 3> forward(
        const Eigen::Tensor<double, 3> &input_image);

    Eigen::Tensor<double, 3> backward(
        const Eigen::Tensor<double, 3> &delta_map);

    void update_params(double learning_rate);

private:

    Eigen::MatrixXd kernel;
    double kernel_bias;

    Eigen::MatrixXd dW;
    double dB;

    Eigen::MatrixXd last_input_image;

    std::random_device rd;
    std::mt19937 gen;

    std::pair <Eigen::MatrixXd, Eigen::MatrixXd> zero_padding(
        const Eigen::MatrixXd &input_image,
        const Eigen::MatrixXd &kernel);

    void Eigen_to_FFTW(
        const Eigen::MatrixXd &eigen_matrix,
        fftw_complex *fftw_array);

    Eigen::MatrixXcd FFTW_to_Eigen(
        fftw_complex *fftw_array,
        const int rows,
        const int cols);

    Eigen::MatrixXcd perform_fft(const Eigen::MatrixXd &input_Matrix);

    Eigen::MatrixXcd multiply_fft_results(
        const Eigen::MatrixXcd &fft_image,
        const Eigen::MatrixXcd &fft_kernel);

    Eigen::MatrixXd perform_ifft(
        const Eigen::MatrixXcd &fft_result);
};