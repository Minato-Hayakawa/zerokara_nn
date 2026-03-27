# zerokara_nn: C++ CNN Implementation from Scratch

A high-performance, from-scratch implementation of a Convolutional Neural Network (CNN) in C++. This project demonstrates the internal mechanics of deep learning by bypassing high-level frameworks and utilizing **Eigen** for linear algebra and **FFTW3** to optimize convolutions via the Fourier Transform.

---

## 🌟 Key Highlights

* **FFT-Accelerated Convolution:** Implements the **Convolution Theorem**, performing 2D convolutions in the frequency domain for mathematical efficiency.
* **Pure C++ Logic:** Every component—from the backpropagation calculus to the MNIST binary parser—is hand-written.
* **Vectorized Computation:** Utilizes `Eigen::Matrix` and `Eigen::Tensor` to ensure high-performance matrix manipulations.
* **Ready for MNIST:** Includes a robust utility suite to load, normalize, and process the classic MNIST handwritten digit dataset.

---

## 🏗 Architecture & Design

### 1. Convolutional Layer (`ConvLayer`)
The flagship feature of this repository. Instead of a sliding window in the spatial domain, this layer:
1.  Performs **Zero-padding** on the input image and kernel.
2.  Executes a **Forward FFT** using the `FFTW3` library.
3.  Applies **Pointwise Multiplication** in the frequency domain.
4.  Retrieves the result via **Inverse FFT**.
5.  Calculates gradients ($dW$, $dB$) using the same Fourier-based approach during backpropagation.

### 2. Dense Layer (`DenseLayer`)
A fully connected layer supporting:
* **Xavier/He Initialization:** Dynamic weight scaling to prevent gradient vanishing/explosion.
* **Efficient Backpropagation:** Optimized gradient flow using Eigen's transposed matrix operations.

### 3. Utility Suite (`Utils`)
* **Activations:** ReLU, Sigmoid, and Softmax.
* **Loss Functions:** Categorical Cross-Entropy Loss.
* **I/O:** Custom binary reader for MNIST `idx` files and OpenCV integration for image preprocessing.

---

## 🔬 Mathematical Foundation

The network optimizes parameters using Gradient Descent. For the `ConvLayer`, the weight updates are derived using the property that convolution in the spatial domain is equivalent to multiplication in the frequency domain:

$$W_{new} = W_{old} - \eta \cdot \mathcal{F}^{-1}(\mathcal{F}(Input) \odot \mathcal{F}(\delta))$$

Where:
* $\mathcal{F}$: Fourier Transform
* $\odot$: Element-wise (Hadamard) product
* $\eta$: Learning rate

---

## 🛠 Tech Stack

| Component | Library |
| :--- | :--- |
| **Linear Algebra** | [Eigen 3](https://eigen.tuxfamily.org/) |
| **FFT Engine** | [FFTW3](http://www.fftw.org/) |
| **Image Processing** | [OpenCV](https://opencv.org/) |
| **Language** | C++20 |

---

## 🚀 Getting Started

### Prerequisites
* CMake (3.10+)
* Eigen 3
* FFTW3
* OpenCV 4.x

### Build & Execution
```bash
# Clone the repository
git clone [https://github.com/Minato-Hayakawa/zerokara_nn.git](https://github.com/Minato-Hayakawa/zerokara_nn.git)

# Run the training
./neural_net.exe
