# MNIST Digit Classification with Artificial Neural Networks (ANN)

This project implements a handwritten digit recognition system using an Artificial Neural Network (ANN) on the classic MNIST dataset. Images are preprocessed using computer vision techniques to enhance feature extraction before feeding them into the network.

## Image Preprocessing Pipeline

Before training, each 28x28 grayscale image passes through the following stages:

1. **Histogram Equalization:** Enhances the contrast of the handwritten digit.
2. **Gaussian Blur:** Smooths the image and reduces pixel-level noise ($5 \times 5$ kernel).
3. **Canny Edge Detection:** Isolates the structural outlines and boundaries of the digits.
4. **Flattening & Normalization:** Converts the 2D processed image into a 1D vector of 784 features and scales pixel values to the $[0, 1]$ range.

## Model Architecture

The Sequential ANN model consists of the following layers:

- **Dense Input Layer:** 128 neurons with ReLU activation.
- **Dropout Layer:** 20% regularization to prevent overfitting.
- **Dense Hidden Layer:** 64 neurons with ReLU activation.
- **Dense Output Layer:** 10 neurons with Softmax activation (representing digits 0-9).

The model is compiled using the `Adam` optimizer and `sparse_categorical_crossentropy` loss function.

## Performance & Results

The model was trained on the **entire MNIST dataset** (60,000 training images, 10,000 test images) for 50 epochs with a batch size of 128.

- **Training Accuracy:** ~97.15%
- **Test (Validation) Accuracy:** **95.65%**
- **Test Loss:** 0.1936

## How to Run

1. Install dependencies:
   ```bash
   pip install tensorflow numpy opencv-python matplotlib
   ```
