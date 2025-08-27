#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include <fftw3.h>

class NeuralNetwork : public Utils{
    private:

        int input_hight,input_width;
        int filter_size;
        int output_hight = input_hight - filter_size + 1;
        int output_width = input_width - filter_size + 1;

    public:

        Eigen::VectorXd dense(
            Eigen::VectorXd &inVector,
            Eigen::VectorXd &outVector,
            void (Utils::*method_ptr)(
                Eigen::VectorXd&,
                Eigen::VectorXd&),
            int units);
        
        Eigen::MatrixXd zero_padding(
            Eigen::MatrixXd &input_image,
            Eigen::MatrixXd &kernel);
        void Eigen_to_FFTW(
            Eigen::MatrixXd &eigen_matrix,
            fftw_complex fftw_array);
        Eigen::MatrixXd FFTW_to_Eigen(
            fftw_complex fftw_array,
            Eigen::MatrixXd &eigen_matrix,
            int rows, int cols);
        Eigen::MatrixXd perfom_fft(Eigen::Matrixxd &input_Matrix);
        Eigen::MatrixXcd multiply_fft_results(
            Eigen::MatrixXcd &fft_image,
            Eigen::MatrixXcd &fft_kernel);
        Eigen::MatrixXd perform_ifft(Eigen::MatrixXcd &fft_result);
};