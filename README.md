# Optimizing Deep Neural Network Performance

Two related projects on making core deep learning operations faster and smaller.

### [`tvm-compiler-optimization/`](tvm-compiler-optimization)
Used Apache TVM to build and accelerate key deep learning operators (1D and 2D depthwise convolutions and matrix multiplication) on CPU and GPU. Starting from baseline implementations, applied hardware-focused optimizations such as tiling, vectorization, and loop unrolling with TVM's scheduling primitives, then benchmarked speedups against unoptimized versions and PyTorch/NumPy references.

### [`dnn-primitives-svd-compression/`](dnn-primitives-svd-compression)
Implemented and evaluated convolution and matrix multiplication techniques for DNNs, including im2col, Winograd, FFT, and SVD, comparing numerical precision against computational efficiency at low precision. The centerpiece applies SVD-based low-rank approximations to compress neural network weight matrices, achieving significant speedups while maintaining high accuracy on MNIST digit recognition, with analysis of accuracy vs. compression trade-offs and runtime.

**Tech:** Python, Apache TVM, PyTorch, NumPy, CUDA

Each subfolder contains its own run instructions.

---

Built as coursework for **ECE 5545: Machine Learning Hardware and Systems** at Cornell Tech (Spring 2025). Starter code, unit tests, and the assignment framework were provided by the course staff; datasets are not included.
