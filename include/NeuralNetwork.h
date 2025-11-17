#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include <fftw3.h>
#include <utility>

class NeuralNetwork : public Utils{

    public:


        std::pair <Eigen::MatrixXd, Eigen::MatrixXd> zero_padding(
            const Eigen::MatrixXd &input_image,
            const Eigen::MatrixXd &kernel);
        void Eigen_to_FFTW(
            const Eigen::MatrixXd &eigen_matrix,
            fftw_complex *fftw_array);
        Eigen::MatrixXcd FFTW_to_Eigen(
            fftw_complex *fftw_array,
            const int rows, const int cols);
        Eigen::MatrixXcd perform_fft(const Eigen::MatrixXd &input_Matrix);
        Eigen::MatrixXcd multiply_fft_results(
            const Eigen::MatrixXcd &fft_image,
            const Eigen::MatrixXcd &fft_kernel);
        Eigen::MatrixXd NeuralNetwork::perform_ifft(
            const Eigen::MatrixXcd &fft_result);
        Eigen::Tensor <double, 3> fft_convolution(
            const Eigen::Tensor <double, 3> &images,
            const Eigen::Tensor <double, 3> &kernels);
};