import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from keras.utils import to_categorical

# ============================================================
# ReLU Activation
# ============================================================

class ReLU:
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        grad = grad_output.copy()
        grad[self.input <= 0] = 0
        return grad

# ============================================================
# Softmax Activation
# ============================================================

class Softmax:
    def forward(self, x):
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.output = exp / np.sum(exp, axis=1, keepdims=True)
        return self.output

    def backward(self, grad_output):
        return grad_output

# ============================================================
# FAST Convolution Layer
# ============================================================

class ConvLayer:
    def __init__(self, num_filters, filter_size, input_depth):

        self.num_filters = num_filters
        self.filter_size = filter_size
        self.input_depth = input_depth

        self.filters = (
            np.random.randn(
                num_filters,
                input_depth,
                filter_size,
                filter_size
            ) * 0.1
        )

    def forward(self, input_data):

        self.input = input_data

        batch_size, depth, height, width = input_data.shape

        out_h = height - self.filter_size + 1
        out_w = width - self.filter_size + 1

        output = np.zeros((batch_size, self.num_filters, out_h, out_w))

        # Faster vectorized convolution
        for f in range(self.num_filters):

            filter_weights = self.filters[f]

            for i in range(out_h):
                for j in range(out_w):

                    region = input_data[
                        :,
                        :,
                        i:i+self.filter_size,
                        j:j+self.filter_size
                    ]

                    output[:, f, i, j] = np.sum(
                        region * filter_weights,
                        axis=(1,2,3)
                    )

        return output

    def backward(self, grad_output, learning_rate):

        batch_size, _, out_h, out_w = grad_output.shape

        grad_filters = np.zeros_like(self.filters)
        grad_input = np.zeros_like(self.input)

        for f in range(self.num_filters):

            for i in range(out_h):
                for j in range(out_w):

                    region = self.input[
                        :,
                        :,
                        i:i+self.filter_size,
                        j:j+self.filter_size
                    ]

                    grad_filters[f] += np.sum(
                        region *
                        grad_output[:, f, i, j][:, None, None, None],
                        axis=0
                    )

                    grad_input[
                        :,
                        :,
                        i:i+self.filter_size,
                        j:j+self.filter_size
                    ] += (
                        self.filters[f] *
                        grad_output[:, f, i, j][:, None, None, None]
                    )

        self.filters -= learning_rate * grad_filters

        return grad_input

# ============================================================
# Max Pooling Layer
# ============================================================

class MaxPool:

    def __init__(self, size=2, stride=2):

        self.size = size
        self.stride = stride

    def forward(self, input_data):

        self.input = input_data

        batch_size, depth, height, width = input_data.shape

        out_h = height // self.size
        out_w = width // self.size

        output = np.zeros((batch_size, depth, out_h, out_w))

        for i in range(out_h):
            for j in range(out_w):

                region = input_data[
                    :,
                    :,
                    i*self.stride:i*self.stride+self.size,
                    j*self.stride:j*self.stride+self.size
                ]

                output[:, :, i, j] = np.max(region, axis=(2,3))

        return output

    def backward(self, grad_output):

        grad_input = np.zeros_like(self.input)

        batch_size, depth, out_h, out_w = grad_output.shape

        for i in range(out_h):
            for j in range(out_w):

                region = self.input[
                    :,
                    :,
                    i*self.stride:i*self.stride+self.size,
                    j*self.stride:j*self.stride+self.size
                ]

                max_region = np.max(region, axis=(2,3), keepdims=True)

                mask = (region == max_region)

                grad_input[
                    :,
                    :,
                    i*self.stride:i*self.stride+self.size,
                    j*self.stride:j*self.stride+self.size
                ] += mask * grad_output[:, :, i, j][:,:,None,None]

        return grad_input

# ============================================================
# Flatten Layer
# ============================================================

class Flatten:

    def forward(self, input_data):

        self.input_shape = input_data.shape

        return input_data.reshape(input_data.shape[0], -1)

    def backward(self, grad_output):

        return grad_output.reshape(self.input_shape)

# ============================================================
# Fully Connected Layer
# ============================================================

class FullyConnected:

    def __init__(self, input_size, output_size):

        self.weights = np.random.randn(
            input_size,
            output_size
        ) * 0.01

        self.biases = np.zeros((1, output_size))

    def forward(self, input_data):

        self.input = input_data

        return np.dot(input_data, self.weights) + self.biases

    def backward(self, grad_output, learning_rate):

        grad_weights = np.dot(self.input.T, grad_output)

        grad_biases = np.sum(
            grad_output,
            axis=0,
            keepdims=True
        )

        grad_input = np.dot(
            grad_output,
            self.weights.T
        )

        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases

        return grad_input

# ============================================================
# Cross Entropy Loss
# ============================================================

class CrossEntropyLoss:

    def forward(self, predictions, labels):

        self.predictions = predictions
        self.labels = labels

        samples = predictions.shape[0]

        predictions = np.clip(predictions, 1e-12, 1-1e-12)

        loss = -np.sum(labels * np.log(predictions)) / samples

        return loss

    def backward(self):

        samples = self.labels.shape[0]

        return (self.predictions - self.labels) / samples

# ============================================================
# Load Dataset
# ============================================================

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

# Reshape
x_train = x_train.reshape(-1, 1, 28, 28)
x_test = x_test.reshape(-1, 1, 28, 28)

# One-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Smaller dataset for fast execution
x_train = x_train[:3000]
y_train = y_train[:3000]

x_test = x_test[:500]
y_test = y_test[:500]

# ============================================================
# CNN Architecture
# ============================================================

conv = ConvLayer(
    num_filters=8,
    filter_size=3,
    input_depth=1
)

relu = ReLU()

pool = MaxPool()

flatten = Flatten()

fc = FullyConnected(
    input_size=8 * 13 * 13,
    output_size=10
)

softmax = Softmax()

loss_function = CrossEntropyLoss()

# ============================================================
# Training Parameters
# ============================================================

learning_rate = 0.01
epochs = 5
batch_size = 128

# ============================================================
# Store Metrics
# ============================================================

loss_history = []
accuracy_history = []

# ============================================================
# Training Loop
# ============================================================

for epoch in range(epochs):

    total_loss = 0
    correct = 0

    for i in range(0, len(x_train), batch_size):

        x_batch = x_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        # Forward
        out = conv.forward(x_batch)
        out = relu.forward(out)
        out = pool.forward(out)
        out = flatten.forward(out)
        out = fc.forward(out)
        out = softmax.forward(out)

        # Loss
        loss = loss_function.forward(out, y_batch)
        total_loss += loss

        predictions = np.argmax(out, axis=1)
        labels = np.argmax(y_batch, axis=1)

        correct += np.sum(predictions == labels)

        # Backward
        grad = loss_function.backward()
        grad = softmax.backward(grad)
        grad = fc.backward(grad, learning_rate)
        grad = flatten.backward(grad)
        grad = pool.backward(grad)
        grad = relu.backward(grad)
        grad = conv.backward(grad, learning_rate)

    accuracy = correct / len(x_train)

    loss_history.append(total_loss)
    accuracy_history.append(accuracy)

    print(f"\nEpoch {epoch+1}/{epochs}")
    print(f"Loss: {total_loss:.4f}")
    print(f"Training Accuracy: {accuracy:.4f}")

# ============================================================
# Testing
# ============================================================

out = conv.forward(x_test)
out = relu.forward(out)
out = pool.forward(out)
out = flatten.forward(out)
out = fc.forward(out)
out = softmax.forward(out)

predictions = np.argmax(out, axis=1)
labels = np.argmax(y_test, axis=1)

test_accuracy = np.mean(predictions == labels)

print("\n===================================")
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
print("===================================")

# ============================================================
# Plot Graphs
# ============================================================

epochs_range = range(1, epochs + 1)

# Epoch vs Loss
plt.figure(figsize=(6,4))
plt.plot(epochs_range, loss_history, marker='o')
plt.title("Epoch vs Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# Epoch vs Accuracy
plt.figure(figsize=(6,4))
plt.plot(epochs_range, accuracy_history, marker='o')
plt.title("Epoch vs Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

# ============================================================
# Display Predictions
# ============================================================

class_names = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
]

plt.figure(figsize=(12,6))

for i in range(10):

    plt.subplot(2,5,i+1)

    plt.imshow(x_test[i][0], cmap='gray')

    plt.title(
        f"Pred: {class_names[predictions[i]]}"
    )

    plt.axis('off')

plt.tight_layout()
plt.show()
