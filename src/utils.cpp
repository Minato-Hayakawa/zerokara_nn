#include "utils.h"


int Utils :: reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

int Utils :: readInt(std::ifstream& file) {
    int val = 0;
    file.read((char*)&val, 4);
    return reverseInt(val);
}

Eigen::Tensor<double, 3> Utils :: load_mnist_images(std::string path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("ファイルが開けません");

    int magic_number = readInt(file);
    int num_images = readInt(file);
    int rows = readInt(file);
    int cols = readInt(file);

    std::cout << "Debug - Images: " << num_images << " Rows: " << rows << " Cols: " << cols << std::endl;

    int limit = 60000; 

    Eigen::Tensor<double, 3> tensor(limit, rows, cols);
    std::vector<unsigned char> buffer(rows * cols);

    for (int i = 0; i < limit; i++) {
        file.read((char*)buffer.data(), rows * cols); // まとめて読み込み
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                tensor(i, r, c) = (double)buffer[r * cols + c] / 255.0;
            }
        }
    }
    
    return tensor;
}

Eigen::VectorXd Utils::load_mnist_labels(std::string path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("ラベルファイルが開けません: " + path);

    int magic_number = readInt(file);
    int num_items = readInt(file);

    std::cout << "Debug Labels - Items: " << num_items << std::endl;

    Eigen::VectorXd labels(num_items);

    for (int i = 0; i < num_items; i++) {
        unsigned char temp = 0;
        file.read((char*)&temp, 1);
        labels(i) = (int)temp; 
    }

    return labels;
}

void Utils::ReLU(const Eigen::VectorXd &inVector, Eigen::VectorXd &outVector){
    outVector=inVector.array().cwiseMax(0.0);
}
void Utils::Sigmoid(const Eigen::VectorXd &inVector, Eigen::VectorXd &outVector){
    outVector = inVector.array().unaryExpr([](double x) {
        return 1.0 / (1.0 + std::exp(-x));
    });
}

void Utils::Softmax(const Eigen::VectorXd &inVector, Eigen::VectorXd &outVector){
    double max_val = inVector.maxCoeff();
    Eigen::ArrayXd exp_values = (inVector.array() - max_val).exp();
    double sum = exp_values.sum();
    outVector = (exp_values / sum).matrix();
}
double Utils::CrossEntropy(
    const Eigen::VectorXd &TargetVector,
    const Eigen::VectorXd &outVector){
    double epsilon = 1e-12;
    Eigen::ArrayXd log_out = Eigen::log(outVector.array() + epsilon);
    return -(TargetVector.array()*log_out).sum();
}

Eigen::VectorXd Utils::output_delta(
    const Eigen::VectorXd &output,
    const Eigen::VectorXd &ground_truth
){
    Eigen::VectorXd error = output - ground_truth;
    Eigen::ArrayXd sigmoid_grad = output.array() * (1.0 - output.array());
    return (error.array() * sigmoid_grad).matrix();
}

Eigen::VectorXd Utils :: output_delta_ReLU(
    const Eigen::VectorXd &output,
    const Eigen::VectorXd &ground_truth
){
    Eigen::VectorXd error = output - ground_truth;
    return error.cwiseProduct((output.array() > 0.0).cast<double>().matrix());
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