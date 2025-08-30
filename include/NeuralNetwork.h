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
                Eigen::VectorXd&,
                Eigen::VectorXd&),
            const int units);
        
        Eigen::MatrixXd zero_padding(
            Eigen::MatrixXd &input_image,
            Eigen::MatrixXd &kernel);
        void Eigen_to_FFTW(
            Eigen::MatrixXd &eigen_matrix,
            fftw_complex *fftw_array);
        Eigen::MatrixXcd FFTW_to_Eigen(
            fftw_complex *fftw_array,
            const int rows, const int cols);
        Eigen::MatrixXd perfom_fft(Eigen::MatrixXd &input_Matrix);
        Eigen::MatrixXcd multiply_fft_results(
            Eigen::MatrixXcd &fft_image,
            Eigen::MatrixXcd &fft_kernel);
        Eigen::MatrixXcd perform_ifft(Eigen::MatrixXcd &fft_result);
        Eigen::MatrixXd fft_convolution(
            Eigen::MatrixXd &image,
            Eigen::MatrixXd &kernel);
};