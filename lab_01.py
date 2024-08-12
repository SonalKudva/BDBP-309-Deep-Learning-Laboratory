import numpy as np
import matplotlib.pyplot as plt

# ReLU function and plot
def plot_relu():
    x = np.linspace(-10, 10, 100)
    relu_output = np.maximum(0, x)

    plt.plot(x, relu_output)
    plt.title("ReLU Activation Function")
    plt.xlabel("x")
    plt.ylabel("ReLU(x)")
    plt.grid(True)
    plt.show()

plot_relu()

# Leaky ReLU function and plot
def plot_leaky_relu(alpha=0.1):
    x = np.linspace(-10, 20, 100)
    leaky_relu_output = np.where(x >= 0, x, alpha * x)

    plt.plot(x, leaky_relu_output)
    plt.title("Leaky ReLU Activation Function")
    plt.xlabel("x")
    plt.ylabel("Leaky ReLU(x)")
    plt.grid(True)
    plt.show()

plot_leaky_relu()

# Softmax function and plot
def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

def plot_softmax(x):
    softmax_output = softmax(x)
    softmax_derivative = softmax_output * (1 - softmax_output)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(x, softmax_output, marker='o')
    plt.title("Softmax Activation Function")
    plt.xlabel("x")
    plt.ylabel("Softmax(x)")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(x, softmax_derivative, marker='o')
    plt.title("Derivative of Softmax")
    plt.xlabel("x")
    plt.ylabel("Softmax'(x)")
    plt.grid(True)

    plt.show()

plot_softmax([2, 3, 4])
plot_softmax([5, 6, 7, 8])

# Tanh function and plot
def plot_tanh():
    x = np.linspace(-10, 10, 50)
    tanh_output = np.tanh(x)
    tanh_derivative = 1 - tanh_output ** 2

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(x, tanh_output, marker='o')
    plt.title("Tanh Activation Function")
    plt.xlabel("x")
    plt.ylabel("tanh(x)")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(x, tanh_derivative, marker='o')
    plt.title("Derivative of Tanh")
    plt.xlabel("x")
    plt.ylabel("tanh'(x)")
    plt.grid(True)

    plt.show()

plot_tanh()

# Sigmoid function and plot
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def plot_sigmoid():
    x = np.linspace(-20, 20, 50)
    sigmoid_output = sigmoid(x)
    sigmoid_derivative = sigmoid_output * (1 - sigmoid_output)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(x, sigmoid_output, marker='o')
    plt.title("Sigmoid Activation Function")
    plt.xlabel("x")
    plt.ylabel("sigmoid(x)")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(x, sigmoid_derivative, marker='o')
    plt.title("Derivative of Sigmoid")
    plt.xlabel("x")
    plt.ylabel("sigmoid'(x)")
    plt.grid(True)

    plt.show()

plot_sigmoid()
