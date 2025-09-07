#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include <fftw3.h>

class NeuralNetwork : public Utils{

    public:

        void dense(
            layer &l,
            Eigen::VectorXd &inVector,
            Eigen::VectorXd &outVector,
            void (Utils::*method_ptr)(
                Eigen::VectorXd &,
                Eigen::VectorXd &));
        
        void dense_backward(
            layer &l,
            const Eigen::VectorXd inVector,
            const Eigen::VectorXd delta,
            Eigen::MatrixXd *dWptr,
            Eigen::VectorXd *dBptr,
            Eigen::VectorXd delta_prev);
        
        Eigen::MatrixXd zero_padding(
            const Eigen::MatrixXd &input_image,
            const Eigen::MatrixXd &kernel);
        void Eigen_to_FFTW(
            const Eigen::MatrixXd &eigen_matrix,
            fftw_complex *fftw_array);
        Eigen::MatrixXcd FFTW_to_Eigen(
            fftw_complex *fftw_array,
            const int rows, const int cols);
        Eigen::MatrixXd perform_fft(const Eigen::MatrixXd &input_Matrix);
        Eigen::MatrixXcd multiply_fft_results(
            const Eigen::MatrixXcd &fft_image,
            const Eigen::MatrixXcd &fft_kernel);
        Eigen::MatrixXcd NeuralNetwork::perform_ifft(
            const Eigen::MatrixXcd &fft_result);
        Eigen::MatrixXcd fft_convolution(
            const Eigen::MatrixXd &image,
            const Eigen::MatrixXd &kernel);
};