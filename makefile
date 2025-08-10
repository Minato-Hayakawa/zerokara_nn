CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2 -Iinclude

SRC = src/main.cpp src/NeuralNetwork.cpp src/Layer.cpp src/utils.cpp
OBJ = $(SRC:.cpp=.o)
TARGET = my_nn

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: all clean