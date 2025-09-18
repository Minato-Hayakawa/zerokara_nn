CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2

OPENCV_FLAGS = $(shell pkg-config --cflags opencv4)
EIGEN_FLAGS = -I/usr/include/eigen3
FFTW_FLAGS = $(shell pkg-config --cflags fftw3)

OPENCV_LIBS = $(shell pkg-config --libs opencv4)
FFTW_LIBS = $(shell pkg-config --libs fftw3)

CXXFLAGS += $(OPENCV_FLAGS) $(EIGEN_FLAGS) $(FFTW_FLAGS)
LDFLAGS = $(OPENCV_LIBS) $(FFTW_LIBS)

SRC = src/main.cpp src/NeuralNetwork.cpp src/layer.cpp src/utils.cpp
OBJ = $(SRC:.cpp=.o)
TARGET = my_nn

.PHONY: all clean

all: $(TARGET)
	@echo "--- Running main program ---"
	./$(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)