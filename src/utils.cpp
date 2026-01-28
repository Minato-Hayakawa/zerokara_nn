#include "utils.h"


int Utils::reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255; c2 = (i >> 8) & 255; c3 = (i >> 16) & 255; c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

Eigen::Tensor<double, 3> Utils :: load_mnist_images(std::string path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("ファイルが開けません");

    int magic_number = 0, num_images = 0, rows = 0, cols = 0;
    file.read((char*)&magic_number, 4);
    file.read((char*)&num_images, 4);
    file.read((char*)&rows, 4);
    file.read((char*)&cols, 4);

    num_images = reverseInt(num_images);
    rows = reverseInt(rows);
    cols = reverseInt(cols);

    int limit = 100; 
    Eigen::Tensor<double, 3> tensor(limit, rows, cols);

    for (int i = 0; i < limit; i++) {
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                unsigned char pixel = 0;
                file.read((char*)&pixel, 1);
                // 0.0 ~ 1.0 に正規化
                tensor(i, r, c) = (double)pixel / 255.0;
            }
        }
    }
    return tensor;
}

void Utils::ReLU(const Eigen::VectorXd &inVector, Eigen::VectorXd &outVector){
    outVector=inVector.array().cwiseMax(0.0);
}
void Utils::Sigmoid(const Eigen::VectorXd &inVector, Eigen::VectorXd &outVector){
    outVector = inVector.array().unaryExpr([](double x) {
        return 1.0 / (1.0 + std::exp(-x));
    });
}

double Utils::CrossEntropy(
    const Eigen::VectorXd &TargetVector,
    const Eigen::VectorXd &outVector){
    double epsilon = 1e-12;
    Eigen::ArrayXd log_out = Eigen::log(outVector.array() + epsilon);
    return -(TargetVector.array()*log_out).sum();
}

Eigen::VectorXd Utils::output_delta(
    const Eigen::VectorXd &y,
    const Eigen::VectorXd &t
){
    return y-t;
}

void Utils::addition(
    const Eigen::VectorXd &inVector1,
    const Eigen::VectorXd &inVector2,
    Eigen::VectorXd &outVector
){
    outVector = inVector1+inVector2;
}
void Utils::multiplication(
    const Eigen::VectorXd &inVector,
    const Eigen::MatrixXd &inMatrix,
    Eigen::VectorXd &outVector
){
    outVector = inMatrix*inVector;
}

void Utils::convert_to_Eigen_tensor(
    const std::vector<cv::Mat> &images,
    Eigen::Tensor<double, 3> &outTensor
){
    outTensor.resize(images.size(), images[0].rows, images[0].cols);
    for (int i = 0; i < images.size(); i++){
        for (int j = 0; j < images[i].rows; j++) {
            for (int k = 0; k < images[i].cols; k++) {
                outTensor(i, j, k) = static_cast<double>(images[i].at<uchar>(j, k)) / 255.0;
            }
        }
    }
}

Eigen::Tensor <double, 3> Utils::load_images(){
    std::string pathpattern = "images/*.jpg;";

    std::vector<std::string> filenames;

    cv::glob(pathpattern, filenames, false);
    std::vector<cv::Mat> loaded_images;

    for (const std::string &filename : filenames){
        cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);;

        if (image.empty()){
            std::cout << "画像ファイルが読み込めませんでした" << filename << std::endl;
            continue;
        }

        loaded_images.push_back(image);
    }

    Eigen::Tensor <double, 3> outTensor;
    Utils::convert_to_Eigen_tensor(loaded_images, outTensor);
    return outTensor;
}

void Utils::convert_tensor_to_matrix(
    const Eigen::Tensor<double, 2> &inTensor,
    Eigen::MatrixXd &outMatrix
){
    outMatrix.resize(inTensor.dimension(0), inTensor.dimension(1));
    for (int i=0; i<inTensor.dimensions()[0]; i++){
        for (int j=0; j<inTensor.dimensions()[1]; j++){
            outMatrix(i, j) = inTensor(i, j);
        }
    }
}

void Utils::convert_matrix_to_tensor(
    const Eigen::MatrixXd &inMatrix,
    Eigen::Tensor<double, 2> &outTensor
){
    outTensor.resize(inMatrix.rows(), inMatrix.cols());
    for (int i=0; i<inMatrix.rows(); i++){
        for (int j=0; j<inMatrix.cols(); j++){
            outTensor(i, j) = inMatrix(i, j);
        }
    }
}

void Utils::convert_matrix_to_vector(
    const Eigen::MatrixXd inMatrix,
    Eigen::VectorXd &outVector
){
    outVector.resize(inMatrix.rows() * inMatrix.cols());
    for (int i=0; i<inMatrix.rows(); i++){
        for (int j=0; j<inMatrix.cols(); j++){
            outVector(i*inMatrix.cols()+j) = inMatrix(i, j);
        }
    }
}