#include "conv_layer.h"

ConvLayer::ConvLayer(const int kernel_size)
:gen(rd())
{
    const double limit = std::sqrt(6.0 / (kernel_size * kernel_size * 2)); // 簡易版
    std::uniform_real_distribution<> d(-limit, limit);

    this -> kernel = Eigen::MatrixXd::NullaryExpr(kernel_size,
                                         kernel_size,
                                         [&]() { return d(gen); });

    this -> kernel_bias = 0.0;
};

std::pair <Eigen::MatrixXd, Eigen::MatrixXd> ConvLayer::zero_padding(
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
    return {padded_image, padded_kernel};
}

void ConvLayer::Eigen_to_FFTW(
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
Eigen::MatrixXcd ConvLayer::FFTW_to_Eigen(
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

Eigen::MatrixXcd ConvLayer::perform_fft(const Eigen::MatrixXd &input_Matrix){
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

Eigen::MatrixXcd ConvLayer::multiply_fft_results(
    const Eigen::MatrixXcd &fft_image,
    const Eigen::MatrixXcd &fft_kernel) {
    return fft_image.cwiseProduct(fft_kernel);
}

Eigen::MatrixXd ConvLayer::perform_ifft(
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

Eigen::Tensor<double, 3> ConvLayer::forward(
    const Eigen::Tensor<double, 3> &input_images)
{
    this -> last_input_images = input_images;

    long num_images = input_images.dimension(0);
    long height = input_images.dimension(1);
    long width = input_images.dimension(2);

    for (int i=0; i<num_images; i++){
        Eigen::Tensor<double, 2> image_chip = input_images.chip(i, 0);
        Eigen::Map<const Eigen::MatrixXd> image_matrix(image_chip.data(), height, width);
        std::pair<Eigen::MatrixXd, Eigen::MatrixXd> padded = zero_padding(image_matrix, this->kernel);
        Eigen::MatrixXcd fft_image = perform_fft(padded.first);
        Eigen::MatrixXcd fft_kernel = perform_fft(padded.second);
        Eigen::MatrixXcd fft_mult = multiply_fft_results(fft_image, fft_kernel);
        Eigen::MatrixXd conv_result = perform_ifft(fft_mult);

        conv_result.array() += this -> kernel_bias;
        Eigen::Tensor<double, 3> output_tensor;
        for (long r = 0; r < height; ++r) {
            for (long c = 0; c < width; ++c) {
                output_tensor(i, r, c) = conv_result(r, c); // (crop処理は省略)
            }
        }
        return output_tensor;
    }
};

void ConvLayer::update_params(double learning_rate){
    this -> kernel -= this -> dW * learning_rate;
    this -> kernel_bias -= this -> dB;
}