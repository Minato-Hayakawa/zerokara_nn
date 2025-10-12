#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include <fftw3.h>

class NeuralNetwork : public Utils{

    public:

        void dense(
            layer &lyrobj,
            Eigen::VectorXd &inVector,
            Eigen::VectorXd &outVector,
            void (Utils::*method_ptr)(
                Eigen::VectorXd &,
                Eigen::VectorXd &));
        
        void dense_backward(
            layer &lyrobj,
            const Eigen::VectorXd &inVector,
            Eigen::VectorXd &delta,
            Eigen::MatrixXd &dW,
            Eigen::VectorXd &dB,
            Eigen::VectorXd &delta_prev);
        
        Eigen::MatrixXd zero_padding(
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