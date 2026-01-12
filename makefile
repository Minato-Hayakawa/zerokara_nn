CXX      := g++
CXXFLAGS := -std=c++14 -O3 -Wall

EIGEN_PATH  := C:/igen-3.4.0/eigen-3.4.0

OPENCV_PATH := C:/opencv/opencv-4.12/build

FFTW_PATH   := C:/fftw/fftw-3.3.5-dll64

INCLUDES := -I. \
            -I$(SRC_DIR) \
            -I$(EIGEN_PATH) \
            -I$(OPENCV_PATH)/include \
            -I$(FFTW_PATH)/include

LIB_DIRS := -L$(OPENCV_PATH)/x64/mingw/lib \
            -L$(FFTW_PATH)

# 注意: OpenCVのバージョンによって数字(460など)を変えてください
# フォルダ内の libopencv_worldXXX.dll.a の名前を確認してください
LIBS     := -lopencv_world460 -lfftw3-3

TARGET   := neural_net.exe

SRCS     := src/main.cpp \
            src/conv_layer.cpp \
            src/dense_layer.cpp \
            src/utils.cpp

OBJS     := $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(LIB_DIRS) $(OBJS) -o $(TARGET) $(LIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	del $(OBJS) $(TARGET)