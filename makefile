CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2

OPENCV_DIR := $(CURDIR)/opencv

EIGEN_DIR := C:/c/eigen-3.4.0

FFTW_FLAGS := $(shell pkg-config --cflags fftw3)
FFTW_LIBS := $(shell pkg-config --libs fftw3)

CXXFLAGS += -Iinclude -I$(OPENCV_DIR)/include -I$(EIGEN_DIR) $(FFTW_FLAGS)

LDFLAGS := -L$(OPENCV_DIR)/lib -lopencv_core -lopencv_highgui -lopencv_imgproc $(FFTW_LIBS)

SRC = $(wildcard src/*.cpp)
OBJ_DIR = obj

OBJ = $(patsubst src/%.cpp,$(OBJ_DIR)/%.o,$(SRC))
TARGET = my_nn

.PHONY: all clean

all: $(TARGET)
	@echo "--- Running main program ---"
	./$(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJ_DIR)/%.o: src/%.cpp | $(OBJ_DIR)
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

clean:
	rm -rf $(OBJ_DIR) $(TARGET)