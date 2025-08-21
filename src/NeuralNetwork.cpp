#include "NeuralNetwork.h"
#include "utils.h"

class NeuralNetwork : public Utils{

        Eigen::VectorXd dense(Eigen::VectorXd inVector,std::string activation,int units){
            Eigen::MatrixXd bias(inVector.size(),units);
            if (activation == "ReLU"){
                return ReLU(multiplication(inVector,bias));
            }else if (activation == "Sigmoid"){
                return Sigmoid(multiplication(inVector,bias));
            }
        }
        Eigen::MatrixXd zero_padding(
            Eigen::MatrixXd input_image,
            Eigen::MatrixXd kernel
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
        }

        void Eigen_to_FFTW(Eigen::MatrixXd eigen_matrix, fftw_complex fftw_array){
            int rows = eigen_matrix.rows();
            int cols = eigen_matrix.cols();

            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    fftw_array[i * cols + j][0] = eigen_matrix(i, j);
                    fftw_array[i * cols + j][1] = 0;                 
                }
            }
        
        Eigen::MatrixXd FFTW_to_Eigen(fftw_complex fftw_array, Eigen::MatrixXd eigen_matrix, int rows, int cols){
            Eigen::MatrixXcd eigen_matrix(rows, cols);
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    eigen_matrix(i, j) = std::complex<double>(fftw_array[i * cols + j][0], fftw_array[i * cols + j][1]);
                }
            }
            return eigen_matrix;
        }

        Eigen::MatrixXd perfom_fft(Eigen::Matrixxd input_Matrix){
            int rows = input_Matrix.rows();
            int cols = input_Matrix.cols();

            fftw_complex* in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * rows * cols);
            fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * rows * cols);
            eigenToFFTW(input_matrix, in);
            fftw_plan plan = fftw_plan_dft_2d(rows, cols, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
            fftw_execute(plan);

            Eigen::MatrixXcd fft_result = fftwToEigen(out, rows, cols);

            fftw_destroy_plan(plan);
            fftw_free(in);
            fftw_free(out);

            return fft_result;
        }

        Eigen::MatrixXcd multiply_fft_results(const Eigen::MatrixXcd& fft_image, const Eigen::MatrixXcd& fft_kernel) {
            return fft_image.cwiseProduct(fft_kernel);
        }
        }

};