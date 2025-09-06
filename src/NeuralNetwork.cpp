#include "NeuralNetwork.h"
#include "utils.h"
#include "layer.h"

void NeuralNetwork::dense(
    layer &l,
    Eigen::VectorXd &inVector,
    Eigen::VectorXd &outVector,
    void (Utils::*method_ptr)(
        Eigen::VectorXd &,
        Eigen::VectorXd &),
    const int units)
    {
    multiplication(inVector,l.weights,outVector);
    addition(l.bias,outVector,outVector);
    (this->*method_ptr)(outVector, outVector);
    }

void dense_backward(
    layer &l,
    const Eigen::VectorXd inVector,
    const Eigen::VectorXd delta,
    Eigen::MatrixXd &dW,
    Eigen::VectorXd &dB,
    Eigen::VectorXd delta_prev
){
    dW = delta * inVector.transpose();
    dB = delta;
    delta_prev = l.weights.transpose() * delta;
}
Eigen::MatrixXd NeuralNetwork::zero_padding(
    const Eigen::MatrixXd &input_image,
    const Eigen::MatrixXd &kernel
){
    int img_rows = input_image.rows();
    int img_cols = input_image.cols();

    int k_rows = kernel.rows();
    int k_cols = kernel.cols();

    int fft_rows = img_rows + k_rows - 1;
    int fft_cols = img_cols + k_cols - 1;

    Eigen::MatrixXd padded_image = Eigen::MatrixXd::Zero(fft_rows, fft_cols);
    Eigen::MatrixXd padded_kernel = Eigen::MatrixXd::Zero(fft_rows, fft_cols);
    padded_image.block(0, 0, img_rows, img_cols) = input_image;
    Eigen::MatrixXd reversed_kernel = kernel.reverse();
    padded_kernel.block(0, 0, k_rows, k_cols) = reversed_kernel;
    return padded_image;
}

void NeuralNetwork::Eigen_to_FFTW(
    const Eigen::MatrixXd &eigen_matrix,
    fftw_complex *fftw_array){
    int rows = eigen_matrix.rows();
    int cols = eigen_matrix.cols();

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fftw_array[i * cols + j][0] = eigen_matrix(i, j);
            fftw_array[i * cols + j][1] = 0;                 
        }
    }
    }
Eigen::MatrixXcd NeuralNetwork::FFTW_to_Eigen(
    fftw_complex *fftw_array,
    const int rows,
    const int cols){
    Eigen::MatrixXcd eigen_matrix(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            eigen_matrix(i, j) = std::complex<double>(fftw_array[i * cols + j][0], fftw_array[i * cols + j][1]);
        }
    }
    return eigen_matrix;
}

Eigen::MatrixXd NeuralNetwork::perform_fft(const Eigen::MatrixXd &input_Matrix){
    int rows = input_Matrix.rows();
    int cols = input_Matrix.cols();

    fftw_complex* in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * rows * cols);
    fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * rows * cols);
    Eigen_to_FFTW(input_Matrix, in);
    fftw_plan plan = fftw_plan_dft_2d(rows, cols, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);

    Eigen::MatrixXcd fft_result = FFTW_to_Eigen(out, rows, cols);

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    return fft_result;
}

Eigen::MatrixXcd NeuralNetwork::multiply_fft_results(
    const Eigen::MatrixXcd &fft_image,
    const Eigen::MatrixXcd &fft_kernel) {
    return fft_image.cwiseProduct(fft_kernel);
}

Eigen::MatrixXcd NeuralNetwork::perform_ifft(
    const Eigen::MatrixXcd &fft_result) {
    int rows = fft_result.rows();
    int cols = fft_result.cols();
    
    fftw_complex* in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * rows * cols);
    fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * rows * cols);
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            in[i * cols + j][0] = fft_result(i, j).real();
            in[i * cols + j][1] = fft_result(i, j).imag();
        }
    }

    fftw_plan plan = fftw_plan_dft_2d(rows, cols, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(plan);

    Eigen::MatrixXd ifft_result(rows, cols);
    double normalization_factor = static_cast<double>(rows * cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            ifft_result(i, j) = out[i * cols + j][0] / normalization_factor;
        }
    }

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    return ifft_result;
}

Eigen::MatrixXcd NeuralNetwork::fft_convolution(
    const Eigen::MatrixXd &image,
    const Eigen::MatrixXd &kernel){
        const Eigen::MatrixXd padded_image = zero_padding(image, kernel);
        const Eigen::MatrixXd padded_kernel = zero_padding(kernel, image);
        const Eigen::MatrixXd fft_image = perform_fft(padded_image);
        const Eigen::MatrixXd fft_kernel = perform_fft(padded_kernel);
        const Eigen::MatrixXcd fft_mult = multiply_fft_results(fft_image, fft_kernel);
        return perform_ifft(fft_mult);
        }
