# Fashion MNIST CNN From Scratch

## Objective

This project implements a Convolutional Neural Network (CNN) completely from scratch using NumPy without using deep learning frameworks like TensorFlow or PyTorch for CNN operations.

The project demonstrates the implementation of:

- Convolution Layer
- ReLU Activation Function
- Max Pooling Layer
- Flatten Layer
- Fully Connected Layer
- Softmax Activation
- Forward Propagation
- Backward Propagation (Backpropagation)
- CNN Training on Fashion MNIST Dataset
- Model Evaluation

---



## Technologies Used

- Python
- NumPy
- Matplotlib
- Keras Dataset Loader

---

## Dataset

The project uses the Fashion MNIST dataset.

Fashion MNIST contains:

- 70,000 grayscale images
- 10 classes of clothing items
- Image size: 28 × 28 pixels

The dataset classes are:

| Label | Class Name |
|------|------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

---

## CNN Architecture

```text
Input Image (28x28)
        ↓
Convolution Layer
        ↓
ReLU Activation
        ↓
Max Pooling Layer
        ↓
Flatten Layer
        ↓
Fully Connected Layer
        ↓
Softmax Output Layer
```

---

## Layers Implemented

### 1. Convolution Layer

The convolution layer extracts important image features such as:

- edges
- textures
- shapes
- patterns

Filters slide over the image and perform convolution operations.

---

### 2. ReLU Activation Function

ReLU introduces non-linearity into the neural network.

Formula:

```text
f(x) = max(0, x)
```

---

### 3. Max Pooling Layer

Pooling reduces the spatial dimensions of feature maps.

Benefits:

- reduces computation
- reduces overfitting
- keeps important features

---

### 4. Flatten Layer

The flatten layer converts 2D feature maps into a 1D vector before passing it to the fully connected layer.

---

### 5. Fully Connected Layer

The fully connected layer performs classification based on extracted features.

---

### 6. Softmax Layer

Softmax converts outputs into probability distributions for classification.

---

## Forward Pass

The forward propagation sequence is:

```text
Input
→ Convolution
→ ReLU
→ MaxPooling
→ Flatten
→ Fully Connected
→ Softmax
→ Prediction
```

---

## Backward Pass

Backpropagation computes gradients and updates weights using gradient descent to minimize loss.

The backward propagation sequence is:

```text
Loss
→ Softmax Gradient
→ Fully Connected Backward
→ Flatten Backward
→ Pooling Backward
→ ReLU Backward
→ Convolution Backward
```

---

## How to Run the Project

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Step 2: Run the Program

```bash
python cnn_from_scratch.py
```

---

## Results

Training Accuracy:58%

Testing Accuracy:62.6%

---

## Author

Name: Ramya R Bijapur  
USN: PES1PG25CS055

---
