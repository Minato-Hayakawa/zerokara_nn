#include "utils.h"

void Utils::ReLU(Eigen::VectorXd &inVector, Eigen::VectorXd &outVector){
    outVector.resize(inVector.size());
    for (int i=0;i<inVector.size();i++){
        outVector[i]=std::max(0.0,inVector[i]);
    }
}
void Utils::Sigmoid(Eigen::VectorXd &inVector, Eigen::VectorXd &outVector){
    outVector.resize(inVector.size());
    for (int i=0;i<inVector.size();i++){
        outVector[i]= 1.0/(1.0 + std::exp(-inVector[i]));
    }
}

double Utils::CrossEntropy(
    Eigen::VectorXd &TargetVector,
    Eigen::VectorXd outVector){
    double epsilon = 1e-12;
    Eigen::ArrayXd log_out = Eigen::log(outVector.array() + epsilon);
    return -(TargetVector.array()*log_out).sum();
}

Eigen::VectorXd Utils::output_delta(
    Eigen::VectorXd &y,
    Eigen::VectorXd &t
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
                outTensor(i, j, k) = static_cast<double>(images[i].at<uchar>(i, j, k));
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
        cv::Mat image = cv::imread(filename);

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

Eigen::MatrixXd Utils::convert_tensor_to_matrix(
    Eigen::Tensor<double, 2> inTensor,
    Eigen::MatrixXd outMatrix
){
    for (int i=0; i<inTensor.dimensions()[0]; i++){
        for (int j=0; j<inTensor.dimensions()[1]; j++){
            outMatrix(i, j) = inTensor(i, j);
        }
    }
}