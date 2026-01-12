CXX      := g++
CXXFLAGS := -std=c++14 -O3 -Wall

EIGEN_PATH  := C:/eigen-3.4.0/eigen-3.4.0

OPENCV_PATH := C:\opencvc\OpenCV-MinGW-Build-OpenCV-4.5.5-x64\OpenCV-MinGW-Build-OpenCV-4.5.5-x64\include

FFTW_PATH   := C:/fftw/fftw-3.3.5-dll64

INCLUDES := -I. \
            -Iinclude \
            -I$(EIGEN_PATH) \
            -I$(OPENCV_PATH)\
            -I$(FFTW_PATH)

LIB_DIRS := -L$(OPENCV_PATH)\x64\mingw\lib\
            -L$(FFTW_PATH)

LIBS     := -lopencv_core455 \
            -lopencv_imgcodecs455 \
            -lopencv_imgproc455 \
            -lopencv_highgui455 \
            -lfftw3-3

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
	del /Q *.o $(TARGET)