cmake_minimum_required(VERSION 3.10)
project(NeuralNetworkProject)

set(CMAKE_CXX_STANDARD 17)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(C:/opencv/opencv/build/include)
include_directories(C:/eigen-3.4.0/eigen-3.4.0)
include_directories(C:/fftw/fftw-3.3.5-dll64)

# Link directories
link_directories(C:/opencv/opencv/build/x64/vc16/lib)
link_directories(C:/fftw/fftw-3.3.5-dll64)

# Source files
set(SOURCES
    src/main.cpp
    src/conv_layer.cpp
    src/dense_layer.cpp
    src/utils.cpp
)

# Main executable
add_executable(neural_network ${SOURCES})
target_link_libraries(neural_network opencv_world4120 libfftw3-3)

# Benchmark executables
add_executable(benchmark_sigmoid src/benchmark_sigmoid.cpp src/conv_layer.cpp src/dense_layer.cpp src/utils.cpp)
target_link_libraries(benchmark_sigmoid opencv_world4120 libfftw3-3)

add_executable(benchmark_softmax src/benchmark_softmax.cpp src/conv_layer.cpp src/dense_layer.cpp src/utils.cpp)
target_link_libraries(benchmark_softmax opencv_world4120 libfftw3-3)

add_executable(benchmark_CNN_sigmoid src/benchmark_CNN+sigmoid.cpp src/conv_layer.cpp src/dense_layer.cpp src/utils.cpp)
target_link_libraries(benchmark_CNN_sigmoid opencv_world4120 libfftw3-3)

add_executable(benchmark_CNN_softmax src/benchmark_CNN+softmax.cpp src/conv_layer.cpp src/dense_layer.cpp src/utils.cpp)
target_link_libraries(benchmark_CNN_softmax opencv_world4120 libfftw3-3)