cmake_minimum_required(VERSION 3.10)
project(NeuralNetworkProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. OpenCV の設定
set(OpenCV_DIR "C:/opencv/opencv/build/x64/vc16/lib")
link_directories("C:/opencv/opencv/build/x64/vc16/lib")
find_package(OpenCV REQUIRED)
# 2. Eigen3 の設定 (インクルードパス)
set(EIGEN3_INCLUDE_DIR "C:/eigen-3.4.0/eigen-3.4.0")

# 3. FFTW3 の設定 (.libファイルのパスとインクルードパス)
set(FFTW3_LIB "C:/fftw/fftw-3.3.5-dll64/libfftw3-3.lib")
set(FFTW3_INCLUDE_DIR "C:/fftw/fftw-3.3.5-dll64")

# 4. ソースファイルの収集
set(SOURCES
    "src/main.cpp"
    "src/conv_layer.cpp"
    "src/dense_layer.cpp"
    "src/utils.cpp"
    # ここに NeuralNetwork.cpp が入っていないことを確認
)

# 5. 実行ファイル作成
add_executable(neural_network ${SOURCES})

# 6. インクルードパスの紐付け (ここを強化しました)
target_include_directories(neural_network PRIVATE 
    "${PROJECT_SOURCE_DIR}/include"
    "${OpenCV_INCLUDE_DIRS}"
    "${EIGEN3_INCLUDE_DIR}"
    "${FFTW3_INCLUDE_DIR}"
)

# 7. リンク
target_link_libraries(neural_network 
    PRIVATE 
    opencv_world4120
    "${FFTW3_LIB}"
)