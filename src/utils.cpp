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

void Utils::cv_to_Eigen(
    std::vector<cv::Mat> loaded_images,
    Eigen::MatrixXd outMat
){
    for (int i = 0; i < loaded_images[i].rows() ++i) {
        for (int j = 0; j < loaded_images[j].cols(); ++j) {
            outMat(i, j) = static_cast<double>(inMat.at<uchar>(i, j));
        }
    }
}

Eigen::Matrix <double, 3, 3> Utils::load_images(std::string name){
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

    Eigen::Matrix<double, 3, 3> outMatrix;
    Utils::cv_to_Eigen(loaded_images, outMatrix);
    return outMatrix
}